NUM_GPUS=1
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_predict1/diffusion/inference/gen3c_single_image.py \
    --checkpoint_dir checkpoints \
    --input_image_path assets/diffusion/000000.png \
    --video_save_name test_single_image_multigpu \
    --num_gpus ${NUM_GPUS} \
    --guidance 1 \
    --offload_tokenizer \
    --offload_text_encoder_model \
    --offload_prompt_upsampler \
    --offload_guardrail_models \
    --disable_guardrail \
    --disable_prompt_encoder \
    --foreground_masking \
    --camera_txt_path /data/Data/combine/traj_raw/shot_000001.txt \
    --camera_txt_extrinsics_type c2w \
    --save_buffer \
    --output_mode 3

# output mode: 1 - point cloud
#              2 - point cloud + video
#              3 - video