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
import hashlib
import os
from pathlib import Path
import multiprocessing as mp
import time
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from moge.model.v1 import MoGeModel
from tqdm.auto import tqdm

from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Buffer
from cosmos_predict1.diffusion.inference.camera_utils import load_camera_trajectory_from_txt
from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
from cosmos_predict1.diffusion.inference.inference_utils import (
    add_common_arguments,
    check_input_frames,
)
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.io import save_video

torch.enable_grad(False)


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _stable_int_seed(base_seed: int, key: str) -> int:
    h = hashlib.sha256(f"{base_seed}|{key}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False)


def _list_images(images_dir: Path) -> list[Path]:
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
    return sorted(files)


def _list_trajectories(trajectories_dir: Path) -> list[Path]:
    files = [p for p in trajectories_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    return sorted(files)


def _rendered_warps_to_numpy_video(all_rendered_warps: list[torch.Tensor]) -> tuple[np.ndarray, int]:
    """Convert collected rendered warp tensors to a visualization video.

    all_rendered_warps: list of tensors each shaped [B, T_chunk, N, C, H, W] on CPU.

    Returns: (video_THWC_uint8, n_max)
    """

    squeezed_warps = [t.squeeze(0) for t in all_rendered_warps]  # (T_chunk, N, C, H, W)
    if not squeezed_warps:
        raise ValueError("No rendered warps to convert")

    n_max = max(t.shape[1] for t in squeezed_warps)

    padded_t_list: list[torch.Tensor] = []
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
    parser = argparse.ArgumentParser(description="GEN3C batch inference (native, no subprocess)")

    add_common_arguments(parser)

    parser.add_argument(
        "--trajectories_dir",
        type=str,
        required=True,
        help="Folder containing camera trajectory txt files (Unity exports).",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Folder containing reference images.",
    )

    # We keep a small subset from gen3c_single_image.py that matter for txt-driven inference.
    parser.add_argument(
        "--prompt_upsampler_dir",
        type=str,
        default="Pixtral-12B",
        help="Prompt upsampler weights directory relative to checkpoint_dir",
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
        "--camera_txt_extrinsics_type",
        type=str,
        choices=["w2c", "c2w"],
        default="c2w",
        help="Interpret extrinsics in the txt trajectory as world-to-camera (w2c) or camera-to-world (c2w).",
    )

    parser.add_argument(
        "--pairing_mode",
        type=str,
        choices=["all", "random_n"],
        default="random_n",
        help=(
            "How to pair trajectories with images: "
            "all=cartesian product (traj x all images), "
            "random_n=for each trajectory, pick N random images (reproducible)."
        ),
    )
    parser.add_argument(
        "--random_n",
        type=int,
        default=2,
        help="When pairing_mode=random_n, pick this many images per trajectory.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help=(
            "Seed for random pairing. If omitted, uses --seed. "
            "Pairing is deterministic per trajectory stem, so different trajectory folders with same filenames "
            "will map to the same images under the same seed."
        ),
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

    parser.add_argument(
        "--output_name_format",
        type=str,
        default="{traj}__{img}__{k}",
        help="Python format for output base name. Available keys: traj,img,k.",
    )

    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help=(
            "Comma-separated CUDA device indices to use for parallel jobs, e.g. '0,1'. "
            "If omitted, uses all visible CUDA devices. If no CUDA, runs on CPU sequentially."
        ),
    )

    parser.add_argument(
        "--jobs_per_gpu",
        type=int,
        default=1,
        help="Max concurrent jobs per GPU (normally keep 1). Total workers = len(devices)*jobs_per_gpu.",
    )

    return parser


def parse_args() -> argparse.Namespace:
    return create_parser().parse_args()


def _validate_batch_args(args: argparse.Namespace) -> None:
    # Keep validation minimal for batch usage; we don't use input_image_or_video_path.
    if args.num_video_frames is None:
        raise ValueError("--num_video_frames must be provided")
    if (args.num_video_frames - 1) % 120 != 0:
        raise ValueError("--num_video_frames must be 121, 241, 361, ... (N*120+1)")


def _predict_moge_depth_from_path(
    current_image_path: str,
    target_h: int,
    target_w: int,
    device: torch.device,
    moge_model: MoGeModel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_image_bgr = cv2.imread(current_image_path)
    if input_image_bgr is None:
        raise FileNotFoundError(f"Input image not found: {current_image_path}")
    input_image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)

    depth_pred_h, depth_pred_w = 720, 1280

    input_image_for_depth_resized = cv2.resize(input_image_rgb, (depth_pred_w, depth_pred_h))
    input_image_for_depth_tensor_chw = torch.tensor(
        input_image_for_depth_resized / 255.0, dtype=torch.float32, device=device
    ).permute(2, 0, 1)

    moge_output_full = moge_model.infer(input_image_for_depth_tensor_chw)
    moge_depth_hw_full = moge_output_full["depth"]
    moge_intrinsics_33_full_normalized = moge_output_full["intrinsics"]
    moge_mask_hw_full = moge_output_full["mask"]

    moge_depth_hw_full = torch.where(
        moge_mask_hw_full == 0, torch.tensor(1000.0, device=moge_depth_hw_full.device), moge_depth_hw_full
    )

    moge_intrinsics_33_full_pixel = moge_intrinsics_33_full_normalized.clone()
    moge_intrinsics_33_full_pixel[0, 0] *= depth_pred_w
    moge_intrinsics_33_full_pixel[1, 1] *= depth_pred_h
    moge_intrinsics_33_full_pixel[0, 2] *= depth_pred_w
    moge_intrinsics_33_full_pixel[1, 2] *= depth_pred_h

    height_scale_factor = target_h / depth_pred_h
    width_scale_factor = target_w / depth_pred_w

    moge_depth_hw = F.interpolate(
        moge_depth_hw_full.unsqueeze(0).unsqueeze(0),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    moge_mask_hw = (
        F.interpolate(
            moge_mask_hw_full.unsqueeze(0).unsqueeze(0).to(torch.float32),
            size=(target_h, target_w),
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
        .to(torch.bool)
    )

    input_image_tensor_chw_target_res = (
        F.interpolate(
            torch.tensor(input_image_rgb / 255.0, dtype=torch.float32, device=device)
            .permute(2, 0, 1)
            .unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
    )

    moge_image_b1chw_float = input_image_tensor_chw_target_res.unsqueeze(0).unsqueeze(1) * 2 - 1

    moge_intrinsics_33 = moge_intrinsics_33_full_pixel.clone()
    moge_intrinsics_33[1, 1] *= height_scale_factor
    moge_intrinsics_33[1, 2] *= height_scale_factor
    moge_intrinsics_33[0, 0] *= width_scale_factor
    moge_intrinsics_33[0, 2] *= width_scale_factor

    moge_depth_b11hw = moge_depth_hw.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    moge_depth_b11hw = torch.nan_to_num(moge_depth_b11hw, nan=1e4)
    moge_depth_b11hw = torch.clamp(moge_depth_b11hw, min=0, max=1e4)

    moge_mask_b11hw = moge_mask_hw.unsqueeze(0).unsqueeze(0).unsqueeze(0)

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
    target_h: int,
    target_w: int,
    device: torch.device,
    moge_model: MoGeModel,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict MoGe depth from an in-memory image tensor.

    image_chw_0_1: (C,H,W) float in [0,1] on device.

    Returns:
        depth_b11hw: (1,1,H,W)
        mask_b11hw: (1,1,H,W) bool
    """

    depth_pred_h, depth_pred_w = 720, 1280

    # Resize to MoGe inference resolution
    img = image_chw_0_1.unsqueeze(0)
    img_resized = F.interpolate(img, size=(depth_pred_h, depth_pred_w), mode="bilinear", align_corners=False).squeeze(0)

    moge_output_full = moge_model.infer(img_resized)
    moge_depth_hw_full = moge_output_full["depth"]
    moge_mask_hw_full = moge_output_full["mask"]

    moge_depth_hw_full = torch.where(
        moge_mask_hw_full == 0, torch.tensor(1000.0, device=moge_depth_hw_full.device), moge_depth_hw_full
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


def _select_pairs(
    traj_files: list[Path],
    image_files: list[Path],
    pairing_mode: str,
    random_n: int,
    base_seed: int,
) -> list[tuple[Path, Path, int]]:
    """Return list of (traj, img, k) pairs.

    k is an integer index within the selected images for the same trajectory.
    """

    if pairing_mode == "all":
        out: list[tuple[Path, Path, int]] = []
        for traj in traj_files:
            for j, img in enumerate(image_files):
                out.append((traj, img, j))
        return out

    if pairing_mode != "random_n":
        raise ValueError(f"Unknown pairing_mode: {pairing_mode}")

    if random_n <= 0:
        raise ValueError("random_n must be > 0")

    if not image_files:
        raise ValueError("No images found")

    out = []
    for traj in traj_files:
        # Use only stem as key so different trajectory folders with same filenames can
        # map to the same images under the same base_seed.
        traj_key = traj.stem
        seed_i = _stable_int_seed(base_seed, traj_key)
        rng = np.random.default_rng(seed_i)

        if random_n <= len(image_files):
            idxs = rng.choice(len(image_files), size=random_n, replace=False)
        else:
            idxs = rng.choice(len(image_files), size=random_n, replace=True)

        for k, idx in enumerate(idxs.tolist()):
            out.append((traj, image_files[int(idx)], k))

    return out


def _run_one_pair(
    *,
    args: argparse.Namespace,
    device: torch.device,
    is_rank0: bool,
    pipeline: Gen3cPipeline,
    moge_model: MoGeModel,
    pair_idx: int,
    total_pairs: int,
    traj_path: Path,
    img_path: Path,
    k: int,
):
    if not check_input_frames(str(img_path), 1):
        log.warning(f"Input image {img_path} is not valid, skipping.")
        return

    log.info(f"[{pair_idx+1}/{total_pairs}] traj={traj_path.name} img={img_path.name}")

    # Load image, predict depth and initialize 3D cache
    (
        moge_image_b1chw_float,
        moge_depth_b11hw,
        _moge_mask_b11hw,
        moge_initial_w2c_b144,
        moge_intrinsics_b133,
    ) = _predict_moge_depth_from_path(str(img_path), args.height, args.width, device, moge_model)

    # Load trajectory
    generated_w2cs, generated_intrinsics = load_camera_trajectory_from_txt(
        file_path=str(traj_path),
        expected_frames=args.num_video_frames,
        device=device,
        extrinsics_type=args.camera_txt_extrinsics_type,
    )

    initial_w2c_for_cache = generated_w2cs[:, 0].clone()
    initial_intrinsics_for_cache = generated_intrinsics[:, 0].clone()

    generator = torch.Generator(device=device).manual_seed(args.seed + pair_idx)

    cache = Cache3D_Buffer(
        frame_buffer_max=pipeline.model.frame_buffer_max,
        generator=generator,
        noise_aug_strength=args.noise_aug_strength,
        input_image=moge_image_b1chw_float[:, 0].clone(),  # [B, C, H, W]
        input_depth=moge_depth_b11hw[:, 0],  # [B, 1, H, W]
        input_w2c=initial_w2c_for_cache,
        input_intrinsics=initial_intrinsics_for_cache,
        filter_points_threshold=args.filter_points_threshold,
        foreground_masking=args.foreground_masking,
    )

    sample_n_frames = pipeline.model.chunk_size

    # Resolve output base
    save_base = args.output_name_format.format(traj=traj_path.stem, img=img_path.stem, k=k)

    need_render_video = args.output_mode in (1, 2)
    need_diffusion_video = args.output_mode in (2, 3)

    # output_mode=3 means "diffusion video only"; ignore --save_buffer to avoid buffer concat.
    if args.output_mode == 3 and args.save_buffer:
        log.warning("output_mode=3 ignores --save_buffer; saving diffusion video only.")
        args.save_buffer = False

    store_rendered_warps = need_render_video or args.save_buffer

    # Mode 1: render-only
    if not need_diffusion_video:
        all_rendered_warps: list[torch.Tensor] = []
        num_ar_iterations = (generated_w2cs.shape[1] - 1) // (sample_n_frames - 1)
        for num_iter in range(num_ar_iterations):
            start_frame_idx = num_iter * (sample_n_frames - 1)
            end_frame_idx = start_frame_idx + sample_n_frames
            current_segment_w2cs = generated_w2cs[:, start_frame_idx:end_frame_idx]
            current_segment_intrinsics = generated_intrinsics[:, start_frame_idx:end_frame_idx]
            rendered_warp_images, _ = cache.render_cache(current_segment_w2cs, current_segment_intrinsics)
            if store_rendered_warps:
                if num_iter == 0:
                    all_rendered_warps.append(rendered_warp_images.clone().cpu())
                else:
                    all_rendered_warps.append(rendered_warp_images[:, 1:].clone().cpu())

        if need_render_video and all_rendered_warps and is_rank0:
            render_video_thwc, n_max = _rendered_warps_to_numpy_video(all_rendered_warps)
            render_save_path = os.path.join(args.video_save_folder, f"{save_base}__render.mp4")
            os.makedirs(os.path.dirname(render_save_path), exist_ok=True)
            save_video(
                video=render_video_thwc,
                fps=args.fps,
                H=args.height,
                W=args.width * n_max,
                video_save_quality=5,
                video_save_path=render_save_path,
            )
            log.info(f"Saved rendered cache video to {render_save_path}")
        return

    # Diffusion generation
    log.info(f"Generating 0 - {sample_n_frames} frames")
    rendered_warp_images, rendered_warp_masks = cache.render_cache(
        generated_w2cs[:, 0:sample_n_frames],
        generated_intrinsics[:, 0:sample_n_frames],
    )

    all_rendered_warps: list[torch.Tensor] = []
    if store_rendered_warps:
        all_rendered_warps.append(rendered_warp_images.clone().cpu())

    generated_output = pipeline.generate(
        prompt=args.prompt,
        image_path=str(img_path),
        negative_prompt=args.negative_prompt,
        rendered_warp_images=rendered_warp_images,
        rendered_warp_masks=rendered_warp_masks,
    )

    if generated_output is None:
        log.critical("Guardrail blocked video2world generation.")
        return

    video, _prompt = generated_output

    num_ar_iterations = (generated_w2cs.shape[1] - 1) // (sample_n_frames - 1)
    for num_iter in range(1, num_ar_iterations):
        start_frame_idx = num_iter * (sample_n_frames - 1)
        end_frame_idx = start_frame_idx + sample_n_frames

        log.info(f"Generating {start_frame_idx} - {end_frame_idx} frames")

        last_frame_hwc_0_255 = torch.tensor(video[-1], device=device)
        pred_image_for_depth_chw_0_1 = last_frame_hwc_0_255.permute(2, 0, 1) / 255.0

        pred_depth, _pred_mask = _predict_moge_depth_from_tensor(
            pred_image_for_depth_chw_0_1, args.height, args.width, device, moge_model
        )

        cache.update_cache(
            new_image=pred_image_for_depth_chw_0_1.unsqueeze(0) * 2 - 1,
            new_depth=pred_depth,
            new_w2c=generated_w2cs[:, start_frame_idx],
            new_intrinsics=generated_intrinsics[:, start_frame_idx],
        )

        current_segment_w2cs = generated_w2cs[:, start_frame_idx:end_frame_idx]
        current_segment_intrinsics = generated_intrinsics[:, start_frame_idx:end_frame_idx]
        rendered_warp_images, rendered_warp_masks = cache.render_cache(current_segment_w2cs, current_segment_intrinsics)

        if store_rendered_warps:
            all_rendered_warps.append(rendered_warp_images[:, 1:].clone().cpu())

        pred_image_for_depth_bcthw_minus1_1 = pred_image_for_depth_chw_0_1.unsqueeze(0).unsqueeze(2) * 2 - 1
        generated_output = pipeline.generate(
            prompt=args.prompt,
            image_path=pred_image_for_depth_bcthw_minus1_1,
            negative_prompt=args.negative_prompt,
            rendered_warp_images=rendered_warp_images,
            rendered_warp_masks=rendered_warp_masks,
        )

        if generated_output is None:
            log.critical("Guardrail blocked video2world generation.")
            break

        video_new, _prompt = generated_output
        video = np.concatenate([video, video_new[1:]], axis=0)

    # Save standalone rendered cache video
    if need_render_video and all_rendered_warps and is_rank0:
        render_video_thwc, n_max = _rendered_warps_to_numpy_video(all_rendered_warps)
        render_save_path = os.path.join(args.video_save_folder, f"{save_base}__render.mp4")
        os.makedirs(os.path.dirname(render_save_path), exist_ok=True)
        save_video(
            video=render_video_thwc,
            fps=args.fps,
            H=args.height,
            W=args.width * n_max,
            video_save_quality=5,
            video_save_path=render_save_path,
        )
        log.info(f"Saved rendered cache video to {render_save_path}")

    final_video_to_save = video
    final_width = args.width

    if args.save_buffer and all_rendered_warps:
        try:
            buffer_numpy_thwc, n_max = _rendered_warps_to_numpy_video(all_rendered_warps)
            final_video_to_save = np.concatenate([buffer_numpy_thwc, final_video_to_save], axis=2)
            final_width = args.width * (1 + n_max)
            log.info(f"Concatenating video with {n_max} warp buffers. Final width={final_width}")
        except Exception as e:
            log.warning(f"Failed to build buffer visualization: {e}")

    if is_rank0:
        video_save_path = os.path.join(args.video_save_folder, f"{save_base}.mp4")
        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        save_video(
            video=final_video_to_save,
            fps=args.fps,
            H=args.height,
            W=final_width,
            video_save_quality=5,
            video_save_path=video_save_path,
        )
        log.info(f"Saved video to {video_save_path}")


def _parse_devices_arg(devices: str | None) -> list[int]:
    if not torch.cuda.is_available():
        return []

    visible = torch.cuda.device_count()
    if devices is None or str(devices).strip() == "":
        return list(range(visible))

    out: list[int] = []
    for part in str(devices).split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= visible:
            raise ValueError(f"Device index {idx} out of range (visible cuda count={visible})")
        out.append(idx)
    if not out:
        raise ValueError("No valid devices parsed from --devices")
    return out


def _worker_main(
    worker_id: int,
    device_index: int | None,
    args: argparse.Namespace,
    jobs: list[tuple[int, str, str, int]],
    stop_event,
    error_queue,
):
    try:
        # Bind this worker to a single GPU (or CPU).
        if device_index is None:
            device = torch.device("cpu")
        else:
            torch.cuda.set_device(device_index)
            device = torch.device(f"cuda:{device_index}")

        log.info(
            f"Worker {worker_id} starting on {device} with {len(jobs)} jobs (single-GPU per job)."
        )

        pipeline = Gen3cPipeline(
            inference_type="video2world",
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name="Gen3C-Cosmos-7B",
            prompt_upsampler_dir=getattr(args, "prompt_upsampler_dir", "Pixtral-12B"),
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

        moge_env_path = os.environ.get("MOGE_MODEL_PATH")
        moge_ckpt_path = (
            Path(moge_env_path)
            if moge_env_path
            else (Path(args.checkpoint_dir) / "moge-vitl" / "model.pt")
        )
        if moge_ckpt_path.exists():
            log.info(f"Worker {worker_id} loading MoGe from local path: {moge_ckpt_path}")
            moge_model = MoGeModel.from_pretrained(moge_ckpt_path).to(device)
        else:
            log.info(f"Worker {worker_id} local MoGe not found; falling back to Ruicheng/moge-vitl")
            moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)

        # Ensure output folder exists.
        os.makedirs(args.video_save_folder, exist_ok=True)

        total_pairs = len(jobs)
        pbar = tqdm(
            total=total_pairs,
            desc=f"worker{worker_id}-dev{device_index if device_index is not None else 'cpu'}",
            position=worker_id,
            leave=False,
        )

        for local_idx, (pair_idx, traj_s, img_s, k) in enumerate(jobs):
            if stop_event.is_set():
                log.warning(f"Worker {worker_id} stopping early due to failure in another worker.")
                break

            traj_path = Path(traj_s)
            img_path = Path(img_s)

            try:
                _run_one_pair(
                    args=args,
                    device=device,
                    is_rank0=True,
                    pipeline=pipeline,
                    moge_model=moge_model,
                    pair_idx=pair_idx,
                    total_pairs=total_pairs,
                    traj_path=traj_path,
                    img_path=img_path,
                    k=k,
                )
            except Exception as e:
                # Any single failure should abort the whole batch.
                stop_event.set()
                tb = traceback.format_exc()
                try:
                    error_queue.put(
                        {
                            "worker_id": worker_id,
                            "device_index": device_index,
                            "job_index": local_idx,
                            "pair_idx": pair_idx,
                            "traj": traj_path.name,
                            "img": img_path.name,
                            "error": repr(e),
                            "traceback": tb,
                        }
                    )
                except Exception:
                    # Best-effort reporting only.
                    pass
                log.exception(
                    f"Worker {worker_id} failed job {local_idx+1}/{total_pairs} (traj={traj_path.name}, img={img_path.name}). Aborting all workers."
                )
                break
            finally:
                pbar.update(1)

        pbar.close()
    except Exception as e:
        # Catch init-time failures (model load, etc.) and abort all workers.
        stop_event.set()
        tb = traceback.format_exc()
        try:
            error_queue.put(
                {
                    "worker_id": worker_id,
                    "device_index": device_index,
                    "job_index": None,
                    "pair_idx": None,
                    "traj": None,
                    "img": None,
                    "error": f"worker_init_failed: {repr(e)}",
                    "traceback": tb,
                }
            )
        except Exception:
            pass
        raise


def main():
    args = parse_args()

    # Make sure prompt-related settings align with typical offline usage.
    if args.prompt is None:
        args.prompt = ""

    # Validate batch-specific args
    _validate_batch_args(args)

    misc.set_random_seed(args.seed)

    trajectories_dir = Path(args.trajectories_dir)
    images_dir = Path(args.images_dir)

    traj_files = _list_trajectories(trajectories_dir)
    image_files = _list_images(images_dir)

    if not traj_files:
        raise FileNotFoundError(f"No txt trajectories found in {trajectories_dir}")
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    base_seed = args.seed if args.random_seed is None else int(args.random_seed)

    pairs = _select_pairs(
        traj_files=traj_files,
        image_files=image_files,
        pairing_mode=args.pairing_mode,
        random_n=args.random_n,
        base_seed=base_seed,
    )

    # This batch runner is job-parallel across GPUs: each job uses a single GPU.
    # So we ignore args.num_gpus>1 context-parallel mode.
    if getattr(args, "num_gpus", 1) != 1:
        log.warning("gen3c_batch_infer.py runs one job per GPU; please set --num_gpus 1 (ignoring).")

    device_indices = _parse_devices_arg(args.devices)
    if not device_indices:
        log.warning("CUDA not available; running sequentially on CPU (very slow).")
        device_indices = [None]

    jobs_per_gpu = max(1, int(args.jobs_per_gpu))
    worker_devices: list[int | None] = []
    for d in device_indices:
        for _ in range(jobs_per_gpu):
            worker_devices.append(d)

    # Pre-pack jobs with stable global pair_idx.
    packed_jobs: list[tuple[int, str, str, int]] = []
    for pair_idx, (traj_path, img_path, k) in enumerate(pairs):
        packed_jobs.append((pair_idx, str(traj_path), str(img_path), int(k)))

    log.info(f"Found {len(traj_files)} trajectories, {len(image_files)} images")
    log.info(
        f"pairing_mode={args.pairing_mode}, total_jobs={len(pairs)}, workers={len(worker_devices)} devices={device_indices}"
    )

    # Split jobs round-robin across workers.
    per_worker: list[list[tuple[int, str, str, int]]] = [[] for _ in range(len(worker_devices))]
    for idx, job in enumerate(packed_jobs):
        per_worker[idx % len(worker_devices)].append(job)

    # Use spawn to be safe with CUDA.
    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    error_queue = ctx.Queue()

    procs: list[mp.Process] = []
    for worker_id, (dev, jobs) in enumerate(zip(worker_devices, per_worker)):
        if not jobs:
            continue
        p = ctx.Process(target=_worker_main, args=(worker_id, dev, args, jobs, stop_event, error_queue))
        p.daemon = False
        p.start()
        procs.append(p)

    # If any worker reports a failure, terminate all remaining workers quickly.
    while True:
        alive = [p for p in procs if p.is_alive()]
        if not alive:
            break

        if stop_event.is_set():
            log.critical("A job failed; terminating all workers...")
            for p in alive:
                try:
                    p.terminate()
                except Exception:
                    pass
            break

        time.sleep(0.2)

    for p in procs:
        p.join()

    # Surface the first error (if any) to the caller with a non-zero exit.
    if not error_queue.empty():
        try:
            first = error_queue.get_nowait()
        except Exception:
            first = None
        if first is not None:
            tb = first.get("traceback")
            if isinstance(tb, str) and len(tb) > 8000:
                tb = tb[-8000:]
            raise RuntimeError(
                "Batch inference aborted due to a failed job: "
                f"worker={first.get('worker_id')} dev={first.get('device_index')} "
                f"traj={first.get('traj')} img={first.get('img')} error={first.get('error')}"
                + (f"\n--- traceback (tail) ---\n{tb}" if tb else "")
            )
        raise RuntimeError("Batch inference aborted due to a failed job.")


if __name__ == "__main__":
    main()
