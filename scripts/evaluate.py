from collections import defaultdict
import contextlib
import os
import copy
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import ddpo_pytorch.prompts
import ddpo_pytorch.rewards
from ddpo_pytorch.stat_tracking import PerPromptStatTracker
from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob, ddim_step_KL
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
import argparse

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)



def evaluate(model_path):
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--resume_from', type=str, default='logs/2024.01.28_01.43.07/checkpoints/checkpoint_60')
    parser.add_argument('--sample_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kl_weight', type=float, default=0.001)
    
    parser.add_argument('--use_lora', type = bool, default=True)
    parser.add_argument('--cfg', type = bool, default=True)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--guidance_scale', type=float, default=5.0)
    parser.add_argument('--pretrained_model', type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument('--pretrained_revision', type=str, default="main")
    parser.add_argument('--out_dir', type=str, default="logs/eval")
    
    parser.add_argument('--prompt_fn', type=str, default="eval_simple_animals")
    parser.add_argument('--reward_fn', type=str, default="aesthetic_score")

    config = parser.parse_args()
    
    if model_path is not None:
        config.resume_from = model_path
    
    config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
    if "checkpoint_" not in os.path.basename(config.resume_from):
        # get the most recent checkpoint in this directory
        checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
        if len(checkpoints) == 0:
            raise ValueError(f"No checkpoints found in {config.resume_from}")
        config.resume_from = os.path.join(
            config.resume_from,
            sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
        )
        
    wandb.init(project="ddpo-evaluation", name=config.resume_from, config=config)

    accelerator = Accelerator()
    set_seed(config.seed, device_specific=True)
    
    pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained_model, revision=config.pretrained_revision)
    
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    
    
    unet_pretrained = copy.deepcopy(pipeline.unet)
    for param in unet_pretrained.parameters():
        param.requires_grad = False
    
    # disable safety checker
    pipeline.safety_checker = None

    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # device = 'cuda'
    inference_dtype = torch.float16
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    unet_pretrained.to(accelerator.device, dtype=inference_dtype)
    
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        # Set correct lora layers
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        pipeline.unet.set_attn_processor(lora_attn_procs)

        # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
        # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
        # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return pipeline.unet(*args, **kwargs)

        unet = _Wrapper(pipeline.unet.attn_processors)
    else:
        unet = pipeline.unet

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            # pipeline.unet.load_attn_procs(input_dir)
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained_model, revision=config.pretrained_revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # prepare prompt and reward fn
    prompt_fn = getattr(ddpo_pytorch.prompts, config.prompt_fn)
    reward_fn = getattr(ddpo_pytorch.rewards, config.reward_fn)()

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.eval_batch_size, 1, 1)


    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    unet = accelerator.prepare(unet)
    executor = futures.ThreadPoolExecutor(max_workers=2)

    print("********** Evaluating **********")
    # print(f"  Sample batch size per device = {config.sample_batch_size}")
    # print(f"  Num processes = {accelerator.num_processes}")

    print(f"Resuming from {config.resume_from}")
    accelerator.load_state(config.resume_from)

    #################### SAMPLING ####################
    pipeline.unet.eval()
    samples = []
    prompts = []
    for i in range(1):
        # generate prompts
        prompts, prompt_metadata = zip(
            *[prompt_fn() for _ in range(config.sample_batch_size)]
        )

        # encode prompts
        prompt_ids = pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
        prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

        # sample
        with autocast():
            images, _, latents, log_probs = pipeline_with_logprob(
                pipeline,
                prompt_embeds=prompt_embeds,  # (batch_size, 77, 768)
                negative_prompt_embeds=sample_neg_prompt_embeds, # (batch_size, 77, 768)
                num_inference_steps=config.num_steps,
                guidance_scale=config.guidance_scale,
                eta=config.eta,
                output_type="pt",
            )
        # list to tensor
        latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
        log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
        timesteps = pipeline.scheduler.timesteps.repeat(config.sample_batch_size, 1)  # (batch_size, num_steps)

        # compute rewards asynchronously
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)
            
        # yield to to make sure reward computation starts
        time.sleep(0)

        samples.append(
            {
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "timesteps": timesteps,
                "latents": latents[:, :-1],  # each entry is the latent before timestep t
                "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                "log_probs": log_probs,
                "rewards": rewards,
            }
        )

    # wait for all rewards to be computed
    for sample in samples:
        rewards, reward_metadata = sample["rewards"].result()
        # accelerator.print(reward_metadata)
        sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)

    # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
    samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

    # # this is a hack to force wandb to log the images as JPEGs instead of PNGs
    images_list = []
    
    for idx, image in enumerate(images):
        pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        pil = pil.resize((256, 256))
        prompt = prompts[idx]
        reward = rewards[idx]
        
        pil.save(config.out_dir +'/'+ f'{prompt}_{reward}.png')
        
        
        images_list.append(wandb.Image(pil, caption=f"{prompt:.25} | {reward:.2f}"))
    
    wandb.log(
        {"images": images_list}
    )
    
    wandb.log({"eval_reward_mean": torch.mean(rewards) ,
               "eval_reward_std": torch.std(rewards) })

    # # gather rewards across processes
    # rewards = accelerator.gather(samples["rewards"]).cpu().numpy()
    
    # rewards += np.random.normal(rewards.shape)*0.1 # get noisy evaluations
    
    # accelerator.log(
    #     {"reward": rewards, "reward_mean": rewards.mean(), "reward_std": rewards.std()}
    # )
    

    # total_batch_size, num_timesteps = samples["timesteps"].shape
    # assert total_batch_size == config.sample_batch_size
    # assert num_timesteps == config.num_steps

    # #################### Calculating KL ####################
    # # shuffle samples along batch dimension
    # perm = torch.randperm(total_batch_size, device=accelerator.device)
    # samples = {k: v[perm] for k, v in samples.items()}

    # # shuffle along time dimension independently for each sample
    # perms = torch.stack(
    #     [torch.randperm(num_timesteps, device=accelerator.device) for _ in range(total_batch_size)]
    # )
    # for key in ["timesteps", "latents", "next_latents", "log_probs"]:
    #     samples[key] = samples[key][torch.arange(total_batch_size, device=accelerator.device)[:, None], perms]

    # # rebatch for training
    # samples_batched = {k: v.reshape(-1, config.eval_batch_size, *v.shape[1:]) for k, v in samples.items()}

    # # dict of lists -> list of dicts for easier iteration
    # samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

    # entropies = list()
    # for i, sample in tqdm(
    #     list(enumerate(samples_batched)),
    #     position=0,
    #     disable=not accelerator.is_local_main_process,
    #     desc="Estimating KL",
    # ):
    #     if config.cfg:
    #         # concat negative prompts to sample prompts to avoid two forward passes
    #         embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
    #     else:
    #         embeds = sample["prompt_embeds"]

    #     KL_sum = 0
    #     for j in tqdm(
    #         range(config.num_steps),
    #         desc="Timestep",
    #         position=1,
    #         leave=False,
    #         disable=not accelerator.is_local_main_process,
    #     ):
    #         with autocast():
    #             if config.cfg:
    #                 noise_pred = unet(
    #                     torch.cat([sample["latents"][:, j]] * 2),
    #                     torch.cat([sample["timesteps"][:, j]] * 2),
    #                     embeds,
    #                 ).sample
    #                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #                 noise_pred = noise_pred_uncond + config.guidance_scale * (
    #                     noise_pred_text - noise_pred_uncond
    #                 )
    #                 # Predicted noise from the pretrained model
    #                 old_noise_pred = unet_pretrained(
    #                     torch.cat([sample["latents"][:, j]] * 2),
    #                     torch.cat([sample["timesteps"][:, j]] * 2),
    #                     embeds,
    #                 ).sample
    #                 old_noise_pred_uncond, old_noise_pred_text = old_noise_pred.chunk(2)
    #                 old_noise_pred = old_noise_pred_uncond + \
    #                     config.guidance_scale * ( old_noise_pred_text - old_noise_pred_uncond )
    #             else:
    #                 noise_pred = unet(
    #                     sample["latents"][:, j],   # (2,4,64,64)
    #                     sample["timesteps"][:, j], # (2,50)
    #                     embeds,  # (4,77,768)
    #                 ).sample
    #                 # Predicted noise from the pretrained model
    #                 old_noise_pred = unet_pretrained(
    #                     sample["latents"][:, j],   # (2,4,64,64)
    #                     sample["timesteps"][:, j], # (2,50)
    #                     embeds,  # (4,77,768)
    #                 ).sample

    #             kl_terms = ddim_step_KL(
    #                 pipeline.scheduler,
    #                 noise_pred,   # (2,4,64,64),
    #                 old_noise_pred, # (2,4,64,64),
    #                 sample["timesteps"][:, j],
    #                 sample["latents"][:, j],
    #                 eta=config.eta,  # 1.0
    #             )
    #             KL_sum += kl_terms.item()
    #     entropies.append(KL_sum)

    # #### Print results ####
    # print("*** Results ***")
    # formatted_rewards = [round(reward, 3) for reward in rewards]
    # print(f"rewards: {formatted_rewards}")
    
    # formatted_entropies = [round(entropy, 3) for entropy in entropies]
    # print(f"KL Entropies: {formatted_entropies}")
    # print(f"KL mean: {np.mean(entropies):.2f}")
    
    # print(f"Regularized reward mean: {np.mean(rewards - config.kl_weight * np.array(entropies)):.2f}")
    



if __name__ == "__main__":
    evaluate(None)
    # model_paths = list()
    
    ### DDPO ###
    # model_paths.append(
    #     'logs/baseline_v8_2023.12.28_05.04.39/checkpoints/checkpoint_98'
    # )  # KL mean: 0.11, Regularized reward mean: -28.23
    
    ### LCB with KL = 1e-2 ###
    
    # model_paths.append(
    #     'logs/LCB_v1/lr=3e-4/kl=1e-2/no-psm_2023.12.29_22.07.46/checkpoints/checkpoint_98'
    # ) # KL mean: 0.06, Regularized reward mean: -69.89
    # model_paths.append(
    #     'logs/LCB_v1/lr=3e-4/kl=1e-2/alpha=1e-4/lambda=1e-1/_2023.12.30_02.04.34/checkpoints/checkpoint_98'
    # ) # KL mean: 0.03, Regularized reward mean: -63.69
    # model_paths.append(
    #     'logs/LCB_v1/lr=3e-4/kl=1e-2/alpha=1e-3/lambda=1e-1/_2023.12.29_21.34.55/checkpoints/checkpoint_98'
    # ) # KL mean: 0.03, Regularized reward mean: -66.04
    # model_paths.append(
    #     './logs/LCB_v1/lr=3e-4/kl=1e-2/alpha=1e-2/lambda=1e-1/_2023.12.28_05.28.36/checkpoints/checkpoint_98'
    # ) # KL mean: 0.03, Regularized reward mean: -63.48
    # model_paths.append(
    #     'logs/LCB_v1/lr=3e-4/kl=1e-2/alpha=1e-1/lambda=1e-1/_2023.12.28_05.17.47/checkpoints/checkpoint_98'
    # ) # KL mean: 0.04, Regularized reward mean: -75.31

    ### LCB with KL = 1e-3 ###
     
    # model_paths.append(
    #     'logs/LCB_v1/lr=3e-4/kl=1e-3/alpha=1e-1/lambda=1e-1/_2023.12.31_21.26.27/checkpoints/checkpoint_98'
    # ) # KL mean: 0.04, Regularized reward mean: -70.94
    # model_paths.append(
    #     'logs/LCB_v1/lr=3e-4/kl=1e-3/alpha=1e-3/lambda=1e-1/_2023.12.31_21.26.40/checkpoints/checkpoint_98'
    # ) # KL mean: 0.08, Regularized reward mean: -26.96
    # model_paths.append(
    #     'logs/LCB_v1/lr=3e-4/kl=1e-3/no-psm_2023.12.31_21.26.36/checkpoints/checkpoint_98'
    # ) # KL mean: 0.14, Regularized reward mean: -49.60
        
    # for path in model_paths:
    #     evaluate(path)
