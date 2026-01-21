# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import cv2
from moge.model.v1 import MoGeModel
import torch
import numpy as np
from pathlib import Path
from cosmos_predict1.diffusion.inference.inference_utils import (
    add_common_arguments,
    check_input_frames,
    validate_args,
)
from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.io import read_prompts_from_file, save_video
from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Buffer
from cosmos_predict1.diffusion.inference.camera_utils import generate_camera_trajectory, load_camera_trajectory_from_txt
import torch.nn.functional as F
torch.enable_grad(False)


def _rendered_warps_to_numpy_video(all_rendered_warps: list[torch.Tensor]) -> tuple[np.ndarray, int]:
    """Convert collected rendered warp tensors to a visualization video.

    Args:
        all_rendered_warps: list of tensors each shaped [B, T_chunk, N, C, H, W] on CPU.

    Returns:
        (video_THWC_uint8, n_max) where the output is stacked horizontally over N buffers.
    """

    squeezed_warps = [t.squeeze(0) for t in all_rendered_warps]  # (T_chunk, N, C, H, W)
    if not squeezed_warps:
        raise ValueError("No rendered warps to convert")

    n_max = max(t.shape[1] for t in squeezed_warps)

    padded_t_list = []
    for sq_t in squeezed_warps:
        current_n_i = sq_t.shape[1]
        padding_needed_dim1 = n_max - current_n_i
        pad_spec = (
            0,
            0,  # W
            0,
            0,  # H
            0,
            0,  # C
            0,
            padding_needed_dim1,  # N
            0,
            0,  # T_chunk
        )
        padded_t = F.pad(sq_t, pad_spec, mode="constant", value=-1.0)
        padded_t_list.append(padded_t)

    full_rendered_warp_tensor = torch.cat(padded_t_list, dim=0)
    t_total, _, c_dim, h_dim, w_dim = full_rendered_warp_tensor.shape
    buffer_video_tchnw = full_rendered_warp_tensor.permute(0, 2, 3, 1, 4)
    buffer_video_tchw_stacked = buffer_video_tchnw.contiguous().view(t_total, c_dim, h_dim, n_max * w_dim)
    buffer_video_tchw_stacked = (buffer_video_tchw_stacked * 0.5 + 0.5) * 255.0
    buffer_numpy_tchw = buffer_video_tchw_stacked.cpu().numpy().astype(np.uint8)
    buffer_numpy_thwc = np.transpose(buffer_numpy_tchw, (0, 2, 3, 1))
    return buffer_numpy_thwc, n_max

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)

    parser.add_argument(
        "--prompt_upsampler_dir",
        type=str,
        default="Pixtral-12B",
        help="Prompt upsampler weights directory relative to checkpoint_dir",
    ) # TODO: do we need this?
    parser.add_argument(
        "--input_image_path",
        type=str,
        help="Input image path for generating a single video",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        choices=[
            "left",
            "right",
            "up",
            "down",
            "zoom_in",
            "zoom_out",
            "clockwise",
            "counterclockwise",
            "none",
        ],
        default="left",
        help="Select a trajectory type from the available options (default: original)",
    )
    parser.add_argument(
        "--camera_rotation",
        type=str,
        choices=["center_facing", "no_rotation", "trajectory_aligned"],
        default="center_facing",
        help="Controls camera rotation during movement: center_facing (rotate to look at center), no_rotation (keep orientation), or trajectory_aligned (rotate in the direction of movement)",
    )
    parser.add_argument(
        "--movement_distance",
        type=float,
        default=0.3,
        help="Distance of the camera from the center of the scene",
    )
    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.0,
        help="Strength of noise augmentation on warped frames",
    )
    parser.add_argument(
        "--save_buffer",
        action="store_true",
        help="If set, save the warped images (buffer) side by side with the output video.",
    )
    parser.add_argument(
        "--filter_points_threshold",
        type=float,
        default=0.05,
        help="If set, filter the points continuity of the warped images.",
    )
    parser.add_argument(
        "--foreground_masking",
        action="store_true",
        help="If set, use foreground masking for the warped images.",
    )
    parser.add_argument(
        "--camera_txt_path",
        type=str,
        default=None,
        help="Path to a txt camera trajectory file. When provided, overrides --trajectory based motion.",
    )
    parser.add_argument(
        "--camera_txt_extrinsics_type",
        type=str,
        choices=["w2c", "c2w"],
        default="w2c",
        help="Interpret extrinsics in the txt trajectory as world-to-camera (w2c) or camera-to-world (c2w).",
    )

    parser.add_argument(
        "--output_mode",
        type=int,
        choices=[1, 2, 3],
        default=2,
        help=(
            "Output saving mode: 1=save rendered 3D-cache video only (skip diffusion video), "
            "2=save rendered 3D-cache video and diffusion video (default), 3=save diffusion video only"
        ),
    )
    return parser

def parse_arguments() -> argparse.Namespace:
    parser = create_parser()
    return parser.parse_args()


def validate_args(args):
    assert args.num_video_frames is not None, "num_video_frames must be provided"
    assert (args.num_video_frames - 1) % 120 == 0, "num_video_frames must be 121, 241, 361, ... (N*120+1)"

def _predict_moge_depth(current_image_path: str | np.ndarray,
                        target_h: int, target_w: int,
                        device: torch.device, moge_model: MoGeModel):
    """Handles MoGe depth prediction for a single image.

    If the image is directly provided as a NumPy array, it should have shape [H, W, C],
    where the channels are RGB and the pixel values are in [0..255].
    """

    if isinstance(current_image_path, str):
        input_image_bgr = cv2.imread(current_image_path)
        if input_image_bgr is None:
            raise FileNotFoundError(f"Input image not found: {current_image_path}")
        input_image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
    else:
        input_image_rgb = current_image_path
    del current_image_path

    depth_pred_h, depth_pred_w = 720, 1280

    input_image_for_depth_resized = cv2.resize(input_image_rgb, (depth_pred_w, depth_pred_h))
    input_image_for_depth_tensor_chw = torch.tensor(input_image_for_depth_resized / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    moge_output_full = moge_model.infer(input_image_for_depth_tensor_chw)
    moge_depth_hw_full = moge_output_full["depth"]
    moge_intrinsics_33_full_normalized = moge_output_full["intrinsics"]
    moge_mask_hw_full = moge_output_full["mask"]

    moge_depth_hw_full = torch.where(moge_mask_hw_full==0, torch.tensor(1000.0, device=moge_depth_hw_full.device), moge_depth_hw_full)
    moge_intrinsics_33_full_pixel = moge_intrinsics_33_full_normalized.clone()
    moge_intrinsics_33_full_pixel[0, 0] *= depth_pred_w
    moge_intrinsics_33_full_pixel[1, 1] *= depth_pred_h
    moge_intrinsics_33_full_pixel[0, 2] *= depth_pred_w
    moge_intrinsics_33_full_pixel[1, 2] *= depth_pred_h

    # Calculate scaling factor for height
    height_scale_factor = target_h / depth_pred_h
    width_scale_factor = target_w / depth_pred_w

    # Resize depth map, mask, and image tensor
    # Resizing depth: (H, W) -> (1, 1, H, W) for interpolate, then squeeze
    moge_depth_hw = F.interpolate(
        moge_depth_hw_full.unsqueeze(0).unsqueeze(0),
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)

    # Resizing mask: (H, W) -> (1, 1, H, W) for interpolate, then squeeze
    moge_mask_hw = F.interpolate(
        moge_mask_hw_full.unsqueeze(0).unsqueeze(0).to(torch.float32),
        size=(target_h, target_w),
        mode='nearest',  # Using nearest neighbor for binary mask
    ).squeeze(0).squeeze(0).to(torch.bool)

    # Resizing image tensor: (C, H, W) -> (1, C, H, W) for interpolate, then squeeze
    input_image_tensor_chw_target_res = F.interpolate(
        input_image_for_depth_tensor_chw.unsqueeze(0),
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)

    moge_image_b1chw_float = input_image_tensor_chw_target_res.unsqueeze(0).unsqueeze(1) * 2 - 1

    moge_intrinsics_33 = moge_intrinsics_33_full_pixel.clone()
    # Adjust intrinsics for resized height
    moge_intrinsics_33[1, 1] *= height_scale_factor  # fy
    moge_intrinsics_33[1, 2] *= height_scale_factor  # cy
    moge_intrinsics_33[0, 0] *= width_scale_factor  # fx
    moge_intrinsics_33[0, 2] *= width_scale_factor  # cx

    moge_depth_b11hw = moge_depth_hw.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    moge_depth_b11hw = torch.nan_to_num(moge_depth_b11hw, nan=1e4)
    moge_depth_b11hw = torch.clamp(moge_depth_b11hw, min=0, max=1e4)
    moge_mask_b11hw = moge_mask_hw.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    # Prepare initial intrinsics [B, 1, 3, 3]
    moge_intrinsics_b133 = moge_intrinsics_33.unsqueeze(0).unsqueeze(0)
    initial_w2c_44 = torch.eye(4, dtype=torch.float32, device=device)
    moge_initial_w2c_b144 = initial_w2c_44.unsqueeze(0).unsqueeze(0)

    return (
        moge_image_b1chw_float,
        moge_depth_b11hw,
        moge_mask_b11hw,
        moge_initial_w2c_b144,
        moge_intrinsics_b133,
    )


def _predict_moge_depth_from_tensor(
    image_chw_0_1: torch.Tensor,
    moge_model: MoGeModel,
    target_h: int = 704,
    target_w: int = 1280,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict MoGe depth from an in-memory image tensor.

    Args:
        image_chw_0_1: (C,H,W) float tensor in [0,1] on the same device as moge_model.
        moge_model: Loaded MoGe model.
        target_h/target_w: Output resolution for depth/mask to match cache update.

    Returns:
        depth_b11hw: (1,1,H,W)
        mask_b11hw: (1,1,H,W) bool
    """

    device = image_chw_0_1.device
    depth_pred_h, depth_pred_w = 720, 1280

    img = image_chw_0_1.unsqueeze(0)
    img_resized = F.interpolate(img, size=(depth_pred_h, depth_pred_w), mode="bilinear", align_corners=False).squeeze(0)

    moge_output_full = moge_model.infer(img_resized)
    moge_depth_hw_full = moge_output_full["depth"]
    moge_mask_hw_full = moge_output_full["mask"]

    moge_depth_hw_full = torch.where(
        moge_mask_hw_full == 0, torch.tensor(1000.0, device=device), moge_depth_hw_full
    )

    depth_hw = F.interpolate(
        moge_depth_hw_full.unsqueeze(0).unsqueeze(0),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    mask_hw = (
        F.interpolate(
            moge_mask_hw_full.unsqueeze(0).unsqueeze(0).to(torch.float32),
            size=(target_h, target_w),
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
        .to(torch.bool)
    )

    depth_b11hw = depth_hw.unsqueeze(0).unsqueeze(0)
    depth_b11hw = torch.nan_to_num(depth_b11hw, nan=1e4)
    depth_b11hw = torch.clamp(depth_b11hw, min=0, max=1e4)

    mask_b11hw = mask_hw.unsqueeze(0).unsqueeze(0)
    return depth_b11hw, mask_b11hw
def demo(args):
    """Run video-to-world generation demo.

    This function handles the main video-to-world generation pipeline, including:
    - Setting up the random seed for reproducibility
    - Initializing the generation pipeline with the provided configuration
    - Processing single or multiple prompts/images/videos from input
    - Generating videos from prompts and images/videos
    - Saving the generated videos and corresponding prompts to disk

    Args:
        cfg (argparse.Namespace): Configuration namespace containing:
            - Model configuration (checkpoint paths, model settings)
            - Generation parameters (guidance, steps, dimensions)
            - Input/output settings (prompts/images/videos, save paths)
            - Performance options (model offloading settings)

    The function will save:
        - Generated MP4 video files
        - Text files containing the processed prompts

    If guardrails block the generation, a critical log message is displayed
    and the function continues to the next prompt if available.
    """
    misc.set_random_seed(args.seed)
    inference_type = "video2world"
    validate_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_rank0 = True

    if args.num_gpus > 1:
        from megatron.core import parallel_state

        from cosmos_predict1.utils import distributed

        distributed.init()
        is_rank0 = distributed.is_rank0()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        process_group = parallel_state.get_context_parallel_group()

    # Initialize video2world generation model pipeline
    pipeline = Gen3cPipeline(
        inference_type=inference_type,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name="Gen3C-Cosmos-7B",
        prompt_upsampler_dir=args.prompt_upsampler_dir,
        enable_prompt_upsampler=not args.disable_prompt_upsampler,
        offload_network=args.offload_diffusion_transformer,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_prompt_upsampler=args.offload_prompt_upsampler,
        offload_guardrail_models=args.offload_guardrail_models,
        disable_guardrail=args.disable_guardrail,
        disable_prompt_encoder=args.disable_prompt_encoder,
        guidance=args.guidance,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_video_frames=121,
        seed=args.seed,
    )

    frame_buffer_max = pipeline.model.frame_buffer_max
    generator = torch.Generator(device=device).manual_seed(args.seed)
    sample_n_frames = pipeline.model.chunk_size
    moge_env_path = os.environ.get("MOGE_MODEL_PATH")
    moge_ckpt_path = Path(moge_env_path) if moge_env_path else (Path(args.checkpoint_dir) / "moge-vitl" / "model.pt")
    if moge_ckpt_path.exists():
        log.info(f"Loading MoGe checkpoint from local path: {moge_ckpt_path}")
        moge_model = MoGeModel.from_pretrained(moge_ckpt_path).to(device)
    else:
        log.info("Local MoGe checkpoint not found; falling back to HuggingFace repo_id Ruicheng/moge-vitl")
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)

    if args.num_gpus > 1:
        pipeline.model.net.enable_context_parallel(process_group)

    # Handle multiple prompts if prompt file is provided
    if args.batch_input_path:
        log.info(f"Reading batch inputs from path: {args.batch_input_path}")
        prompts = read_prompts_from_file(args.batch_input_path)
    else:
        # Single prompt case
        prompts = [{"prompt": args.prompt, "visual_input": args.input_image_path}]

    os.makedirs(os.path.dirname(args.video_save_folder), exist_ok=True)
    for i, input_dict in enumerate(prompts):
        current_prompt = input_dict.get("prompt", None)
        if current_prompt is None and args.disable_prompt_upsampler:
            log.critical("Prompt is missing, skipping world generation.")
            continue
        current_image_path = input_dict.get("visual_input", None)
        if current_image_path is None:
            log.critical("Visual input is missing, skipping world generation.")
            continue

        # Check input frames
        if not check_input_frames(current_image_path, 1):
            print(f"Input image {current_image_path} is not valid, skipping.")
            continue

        # load image, predict depth and initialize 3D cache
        (
            moge_image_b1chw_float,
            moge_depth_b11hw,
            moge_mask_b11hw,
            moge_initial_w2c_b144,
            moge_intrinsics_b133,
        ) = _predict_moge_depth(
            current_image_path, args.height, args.width, device, moge_model
        )

        initial_w2c_for_cache = moge_initial_w2c_b144[:, 0].clone()
        initial_intrinsics_for_cache = moge_intrinsics_b133[:, 0].clone()

        use_custom_trajectory = args.camera_txt_path is not None

        if use_custom_trajectory:
            try:
                generated_w2cs, generated_intrinsics = load_camera_trajectory_from_txt(
                    file_path=args.camera_txt_path,
                    expected_frames=args.num_video_frames,
                    device=device,
                    extrinsics_type=args.camera_txt_extrinsics_type,
                )
            except (ValueError, FileNotFoundError) as e:
                log.critical(f"Failed to load external camera trajectory: {e}")
                continue

            initial_w2c_for_cache = generated_w2cs[:, 0].clone()
            initial_intrinsics_for_cache = generated_intrinsics[:, 0].clone()
        else:
            initial_cam_w2c_for_traj = initial_w2c_for_cache[0]
            initial_cam_intrinsics_for_traj = initial_intrinsics_for_cache[0]
            try:
                generated_w2cs, generated_intrinsics = generate_camera_trajectory(
                    trajectory_type=args.trajectory,
                    initial_w2c=initial_cam_w2c_for_traj,
                    initial_intrinsics=initial_cam_intrinsics_for_traj,
                    num_frames=args.num_video_frames,
                    movement_distance=args.movement_distance,
                    camera_rotation=args.camera_rotation,
                    center_depth=1.0,
                    device=device.type,
                )
            except (ValueError, NotImplementedError) as e:
                log.critical(f"Failed to generate trajectory: {e}")
                continue

        cache = Cache3D_Buffer(
            frame_buffer_max=frame_buffer_max,
            generator=generator,
            noise_aug_strength=args.noise_aug_strength,
            input_image=moge_image_b1chw_float[:, 0].clone(), # [B, C, H, W]
            input_depth=moge_depth_b11hw[:, 0],       # [B, 1, H, W]
            # input_mask=moge_mask_b11hw[:, 0],         # [B, 1, H, W]
            input_w2c=initial_w2c_for_cache,
            input_intrinsics=initial_intrinsics_for_cache,
            filter_points_threshold=args.filter_points_threshold,
            foreground_masking=args.foreground_masking,
        )

        # Resolve the base name used for saving outputs.
        save_base = f"{i}" if args.batch_input_path else args.video_save_name

        need_render_video = args.output_mode in (1, 2)
        need_diffusion_video = args.output_mode in (2, 3)

        # output_mode=3 means "diffusion video only". Make it take precedence over --save_buffer
        # to avoid concatenating warp/buffer visualizations.
        if args.output_mode == 3 and args.save_buffer:
            log.warning("output_mode=3 ignores --save_buffer; saving diffusion video only.")
            args.save_buffer = False

        # Collect rendered warps if we either need a standalone render video, or we want to
        # concatenate buffers into the final diffusion output via --save_buffer.
        store_rendered_warps = need_render_video or args.save_buffer

        # Mode 1: render 3D cache only (no diffusion generation).
        if not need_diffusion_video:
            all_rendered_warps = []
            num_ar_iterations = (generated_w2cs.shape[1] - 1) // (sample_n_frames - 1)
            for num_iter in range(num_ar_iterations):
                start_frame_idx = num_iter * (sample_n_frames - 1)
                end_frame_idx = start_frame_idx + sample_n_frames
                log.info(f"Rendering {start_frame_idx} - {end_frame_idx} frames")
                current_segment_w2cs = generated_w2cs[:, start_frame_idx:end_frame_idx]
                current_segment_intrinsics = generated_intrinsics[:, start_frame_idx:end_frame_idx]
                rendered_warp_images, _ = cache.render_cache(current_segment_w2cs, current_segment_intrinsics)
                if store_rendered_warps:
                    if num_iter == 0:
                        all_rendered_warps.append(rendered_warp_images.clone().cpu())
                    else:
                        all_rendered_warps.append(rendered_warp_images[:, 1:].clone().cpu())

            if need_render_video and all_rendered_warps:
                render_video_thwc, n_max = _rendered_warps_to_numpy_video(all_rendered_warps)
                render_save_path = os.path.join(args.video_save_folder, f"{save_base}__render.mp4")
                if is_rank0:
                    save_video(
                        video=render_video_thwc,
                        fps=args.fps,
                        H=args.height,
                        W=args.width * n_max,
                        video_save_quality=5,
                        video_save_path=render_save_path,
                    )
                    log.info(f"Saved rendered cache video to {render_save_path}")
            continue

        log.info(f"Generating 0 - {sample_n_frames} frames")
        rendered_warp_images, rendered_warp_masks = cache.render_cache(
            generated_w2cs[:, 0:sample_n_frames],
            generated_intrinsics[:, 0:sample_n_frames],
        )

        all_rendered_warps = []
        if store_rendered_warps:
            all_rendered_warps.append(rendered_warp_images.clone().cpu())

        # Generate video
        generated_output = pipeline.generate(
            prompt=current_prompt,
            image_path=current_image_path,
            negative_prompt=args.negative_prompt,
            rendered_warp_images=rendered_warp_images,
            rendered_warp_masks=rendered_warp_masks,
        )
        if generated_output is None:
            log.critical("Guardrail blocked video2world generation.")
            continue
        video, prompt = generated_output

        num_ar_iterations = (generated_w2cs.shape[1] - 1) // (sample_n_frames - 1)
        for num_iter in range(1, num_ar_iterations):
            start_frame_idx = num_iter * (sample_n_frames - 1) # Overlap by 1 frame
            end_frame_idx = start_frame_idx + sample_n_frames

            log.info(f"Generating {start_frame_idx} - {end_frame_idx} frames")

            last_frame_hwc_0_255 = torch.tensor(video[-1], device=device)
            pred_image_for_depth_chw_0_1 = last_frame_hwc_0_255.permute(2, 0, 1) / 255.0 # (C,H,W), range [0,1]

            pred_depth, pred_mask = _predict_moge_depth_from_tensor(
                pred_image_for_depth_chw_0_1,
                moge_model,
                target_h=args.height,
                target_w=args.width,
            )

            cache.update_cache(
                new_image=pred_image_for_depth_chw_0_1.unsqueeze(0) * 2 - 1, # (B,C,H,W) range [-1,1]
                new_depth=pred_depth, #  (1,1,H,W)
                # new_mask=pred_mask,   # (1,1,H,W)
                new_w2c=generated_w2cs[:, start_frame_idx],
                new_intrinsics=generated_intrinsics[:, start_frame_idx],
            )
            current_segment_w2cs = generated_w2cs[:, start_frame_idx:end_frame_idx]
            current_segment_intrinsics = generated_intrinsics[:, start_frame_idx:end_frame_idx]
            rendered_warp_images, rendered_warp_masks = cache.render_cache(
                current_segment_w2cs,
                current_segment_intrinsics,
            )

            if store_rendered_warps:
                all_rendered_warps.append(rendered_warp_images[:, 1:].clone().cpu())

            pred_image_for_depth_bcthw_minus1_1 = (
                pred_image_for_depth_chw_0_1.unsqueeze(0).unsqueeze(2) * 2 - 1
            )  # (B,C,T,H,W), range [-1,1]
            generated_output = pipeline.generate(
                prompt=current_prompt,
                image_path=pred_image_for_depth_bcthw_minus1_1,
                negative_prompt=args.negative_prompt,
                rendered_warp_images=rendered_warp_images,
                rendered_warp_masks=rendered_warp_masks,
            )
            if generated_output is None:
                log.critical("Guardrail blocked video2world generation.")
                break
            video_new, prompt = generated_output
            video = np.concatenate([video, video_new[1:]], axis=0)

        # Save standalone rendered cache video if requested.
        if need_render_video and all_rendered_warps:
            render_video_thwc, n_max = _rendered_warps_to_numpy_video(all_rendered_warps)
            render_save_path = os.path.join(args.video_save_folder, f"{save_base}__render.mp4")
            if is_rank0:
                save_video(
                    video=render_video_thwc,
                    fps=args.fps,
                    H=args.height,
                    W=args.width * n_max,
                    video_save_quality=5,
                    video_save_path=render_save_path,
                )
                log.info(f"Saved rendered cache video to {render_save_path}")

        # Final video processing
        final_video_to_save = video
        final_width = args.width

        if args.save_buffer and all_rendered_warps:
            try:
                buffer_numpy_thwc, n_max = _rendered_warps_to_numpy_video(all_rendered_warps)
                final_video_to_save = np.concatenate([buffer_numpy_thwc, final_video_to_save], axis=2)
                final_width = args.width * (1 + n_max)
                log.info(
                    f"Concatenating video with {n_max} warp buffers. Final video width will be {final_width}"
                )
            except Exception as e:
                log.warning(f"Failed to build buffer visualization: {e}")


        video_save_path = os.path.join(args.video_save_folder, f"{save_base}.mp4")

        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)

        # Save video
        if is_rank0:
            save_video(
                video=final_video_to_save,
                fps=args.fps,
                H=args.height,
                W=final_width,
                video_save_quality=5,
                video_save_path=video_save_path,
            )
            log.info(f"Saved video to {video_save_path}")

    # clean up properly
    if args.num_gpus > 1:
        parallel_state.destroy_model_parallel()
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_arguments()
    if args.prompt is None:
        args.prompt = ""
    args.disable_guardrail = True
    args.disable_prompt_upsampler = True
    demo(args)
