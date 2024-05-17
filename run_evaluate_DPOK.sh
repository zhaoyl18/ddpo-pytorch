base_dir="logs_DPOK/2024.01.28_01.43.07/checkpoints"

# Iterate through every 3rd checkpoint
for i in {0..60..3}
do
  # Construct checkpoint path
  checkpoint_path="$base_dir/checkpoint_$i"
  
  # Run the command with the constructed checkpoint path
  CUDA_VISIBLE_DEVICES=5 python scripts/evaluate.py --resume_from $checkpoint_path --run_name Plot-Eval/DPOK/ckpt_$i --num_samples 128
done
