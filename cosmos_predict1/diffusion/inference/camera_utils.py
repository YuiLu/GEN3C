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

import torch
import math
from pathlib import Path
import torch.nn.functional as F
from .forward_warp_utils_pytorch import unproject_points

def apply_transformation(Bx4x4, another_matrix):
    B = Bx4x4.shape[0]
    if another_matrix.dim() == 2:
        another_matrix = another_matrix.unsqueeze(0).expand(B, -1, -1)  # Make another_matrix compatible with batch size
    transformed_matrix = torch.bmm(Bx4x4, another_matrix)  # Shape: (B, 4, 4)

    return transformed_matrix


def look_at_matrix(camera_pos, target, invert_pos=True):
    """Creates a 4x4 look-at matrix, keeping the camera pointing towards a target."""
    forward = (target - camera_pos).float()
    forward = forward / torch.norm(forward)

    up = torch.tensor([0.0, 1.0, 0.0], device=camera_pos.device)  # assuming Y-up coordinate system
    right = torch.cross(up, forward)
    right = right / torch.norm(right)
    up = torch.cross(forward, right)

    look_at = torch.eye(4, device=camera_pos.device)
    look_at[0, :3] = right
    look_at[1, :3] = up
    look_at[2, :3] = forward
    look_at[:3, 3] = (-camera_pos) if invert_pos else camera_pos

    return look_at

def create_horizontal_trajectory(
    world_to_camera_matrix, center_depth, positive=True, n_steps=13, distance=0.1, device="cuda", axis="x", camera_rotation="center_facing"
):
    look_at = torch.tensor([0.0, 0.0, center_depth]).to(device)
    # Spiral motion key points
    trajectory = []
    translation_positions = []
    initial_camera_pos = torch.tensor([0, 0, 0], device=device)

    for i in range(n_steps):
        if axis == "x": # pos - right
            x = i * distance * center_depth / n_steps * (1 if positive else -1)
            y = 0
            z = 0
        elif axis == "y": # pos - down
            x = 0
            y = i * distance * center_depth / n_steps * (1 if positive else -1)
            z = 0
        elif axis == "z": # pos - in
            x = 0
            y = 0
            z = i * distance * center_depth / n_steps * (1 if positive else -1)
        else:
            raise ValueError("Axis should be x, y or z")

        translation_positions.append(torch.tensor([x, y, z], device=device))

    for pos in translation_positions:
        camera_pos = initial_camera_pos + pos
        if camera_rotation == "trajectory_aligned":
            _look_at = look_at + pos * 2
        elif camera_rotation == "center_facing":
            _look_at = look_at
        elif camera_rotation == "no_rotation":
            _look_at = look_at + pos
        else:
            raise ValueError("Camera rotation should be center_facing or trajectory_aligned")
        view_matrix = look_at_matrix(camera_pos, _look_at)
        trajectory.append(view_matrix)
    trajectory = torch.stack(trajectory)
    return apply_transformation(trajectory, world_to_camera_matrix)


def create_spiral_trajectory(
    world_to_camera_matrix,
    center_depth,
    radius_x=0.03,
    radius_y=0.02,
    radius_z=0.0,
    positive=True,
    camera_rotation="center_facing",
    n_steps=13,
    device="cuda",
    start_from_zero=True,
    num_circles=1,
):

    look_at = torch.tensor([0.0, 0.0, center_depth]).to(device)

    # Spiral motion key points
    trajectory = []
    spiral_positions = []
    initial_camera_pos = torch.tensor([0, 0, 0], device=device)  # world_to_camera_matrix[:3, 3].clone()

    example_scale = 1.0

    theta_max = 2 * math.pi * num_circles

    for i in range(n_steps):
        # theta = 2 * math.pi * i / (n_steps-1)  # angle for each point
        theta = theta_max * i / (n_steps - 1)  # angle for each point
        if start_from_zero:
            x = radius_x * (math.cos(theta) - 1) * (1 if positive else -1) * (center_depth / example_scale)
        else:
            x = radius_x * (math.cos(theta)) * (center_depth / example_scale)

        y = radius_y * math.sin(theta) * (center_depth / example_scale)
        z = radius_z * math.sin(theta) * (center_depth / example_scale)
        spiral_positions.append(torch.tensor([x, y, z], device=device))

    for pos in spiral_positions:
        if camera_rotation == "center_facing":
            view_matrix = look_at_matrix(initial_camera_pos + pos, look_at)
        elif camera_rotation == "trajectory_aligned":
            view_matrix = look_at_matrix(initial_camera_pos + pos, look_at + pos * 2)
        elif camera_rotation == "no_rotation":
            view_matrix = look_at_matrix(initial_camera_pos + pos, look_at + pos)
        else:
            raise ValueError("Camera rotation should be center_facing, trajectory_aligned or no_rotation")
        trajectory.append(view_matrix)
    trajectory = torch.stack(trajectory)
    return apply_transformation(trajectory, world_to_camera_matrix)


def generate_camera_trajectory(
    trajectory_type: str,
    initial_w2c: torch.Tensor,  # Shape: (4, 4)
    initial_intrinsics: torch.Tensor,  # Shape: (3, 3)
    num_frames: int,
    movement_distance: float,
    camera_rotation: str,
    center_depth: float = 1.0,
    device: str = "cuda",
):
    """
    Generates a sequence of camera poses (world-to-camera matrices) and intrinsics
    for a specified trajectory type.

    Args:
        trajectory_type: Type of trajectory (e.g., "left", "right", "up", "down", "zoom_in", "zoom_out").
        initial_w2c: Initial world-to-camera matrix (4x4 tensor or num_framesx4x4 tensor).
        initial_intrinsics: Camera intrinsics matrix (3x3 tensor or num_framesx3x3 tensor).
        num_frames: Number of frames (steps) in the trajectory.
        movement_distance: Distance factor for the camera movement.
        camera_rotation: Type of camera rotation ('center_facing', 'no_rotation', 'trajectory_aligned').
        center_depth: Depth of the center point the camera might focus on.
        device: Computation device ("cuda" or "cpu").

    Returns:
        A tuple (generated_w2cs, generated_intrinsics):
        - generated_w2cs: Batch of world-to-camera matrices for the trajectory (1, num_frames, 4, 4 tensor).
        - generated_intrinsics: Batch of camera intrinsics for the trajectory (1, num_frames, 3, 3 tensor).
    """
    if trajectory_type in ["clockwise", "counterclockwise"]:
        new_w2cs_seq = create_spiral_trajectory(
            world_to_camera_matrix=initial_w2c,
            center_depth=center_depth,
            n_steps=num_frames,
            positive=trajectory_type == "clockwise",
            device=device,
            camera_rotation=camera_rotation,
            radius_x=movement_distance,
            radius_y=movement_distance,
        )
    else:
        if trajectory_type == "left":
            positive = False
            axis = "x"
        elif trajectory_type == "right":
            positive = True
            axis = "x"
        elif trajectory_type == "up":
            positive = False  # Assuming 'up' means camera moves in negative y direction if y points down
            axis = "y"
        elif trajectory_type == "down":
            positive = True # Assuming 'down' means camera moves in positive y direction if y points down
            axis = "y"
        elif trajectory_type == "zoom_in":
            positive = True  # Assuming 'zoom_in' means camera moves in positive z direction (forward)
            axis = "z"
        elif trajectory_type == "zoom_out":
            positive = False # Assuming 'zoom_out' means camera moves in negative z direction (backward)
            axis = "z"
        else:
            raise ValueError(f"Unsupported trajectory type: {trajectory_type}")

        # Generate world-to-camera matrices using create_horizontal_trajectory
        new_w2cs_seq = create_horizontal_trajectory(
            world_to_camera_matrix=initial_w2c,
            center_depth=center_depth,
            n_steps=num_frames,
            positive=positive,
            axis=axis,
            distance=movement_distance,
            device=device,
            camera_rotation=camera_rotation,
        )

    generated_w2cs = new_w2cs_seq.unsqueeze(0)  # Shape: [1, num_frames, 4, 4]
    if initial_intrinsics.dim() == 2:
        generated_intrinsics = initial_intrinsics.unsqueeze(0).unsqueeze(0).repeat(1, num_frames, 1, 1)
    else:
        generated_intrinsics = initial_intrinsics.unsqueeze(0)

    return generated_w2cs, generated_intrinsics


def _invert_extrinsics(rotation: torch.Tensor, translation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert camera-to-world extrinsics to world-to-camera."""
    rot_inv = rotation.transpose(0, 1)
    trans_inv = -rot_inv @ translation
    return rot_inv, trans_inv


def load_camera_trajectory_from_txt(
    file_path: str,
    expected_frames: int | None = None,
    device: torch.device | str = "cuda",
    extrinsics_type: str = "w2c",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load per-frame camera extrinsics and intrinsics from a whitespace separated txt file.

    Each non-comment line should contain at least 19 values in the following order:
        timestamp fx fy cx cy extra1 extra2
        R11 R12 R13 tx
        R21 R22 R23 ty
        R31 R32 R33 tz

    Additional trailing values are ignored. Lines starting with ``#`` or ``//`` are skipped.

    Args:
        file_path: Path to the txt file containing the trajectory.
    expected_frames: Optional number of frames to match by interpolation or sampling.
    device: Target device for the returned tensors.
    extrinsics_type: Interpret the rotation/translation as ``w2c`` (default) or ``c2w``.

    Notes:
        GEN3C's forward-warp / unprojection utilities assume an OpenCV-like camera convention
        (image x right, y down, camera z forward). Many Unity exports are authored in a Y-up
        camera convention. This loader therefore converts incoming extrinsics to the convention
        expected by the inference pipeline.

    Returns:
        Tuple of tensors ``(w2c, intrinsics)`` with shapes ``[1, T, 4, 4]`` and ``[1, T, 3, 3]`` respectively.
    """

    device = torch.device(device)
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Camera trajectory file not found: {path}")

    w2c_mats: list[torch.Tensor] = []
    intrinsic_mats: list[torch.Tensor] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("//"):
                continue

            # Remove trailing comments inline
            for comment_token in ("#", "//"):
                if comment_token in line:
                    line = line.split(comment_token, 1)[0].strip()
            if not line:
                continue

            tokens = line.replace(",", " ").split()
            if len(tokens) < 19:
                # Skip headers or malformed lines silently so long as no cameras were parsed yet.
                if w2c_mats:
                    raise ValueError(
                        f"Line {line_idx + 1} in {path} has {len(tokens)} values; 19 required."
                    )
                else:
                    continue

            try:
                values = [float(tok) for tok in tokens[:19]]
            except ValueError as exc:
                if w2c_mats:
                    raise ValueError(
                        f"Failed to parse numeric values on line {line_idx + 1} of {path}: {exc}"
                    ) from exc
                continue

            # Unpack values
            _, fx, fy, cx, cy, *_unused = values[:7]
            r11, r12, r13, tx = values[7:11]
            r21, r22, r23, ty = values[11:15]
            r31, r32, r33, tz = values[15:19]

            rotation = torch.tensor(
                [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
                dtype=torch.float32,
                device=device,
            )
            translation = torch.tensor([tx, ty, tz], dtype=torch.float32, device=device)

            if extrinsics_type not in {"w2c", "c2w"}:
                raise ValueError("extrinsics_type must be either 'w2c' or 'c2w'")

            # Convert to world-to-camera first.
            if extrinsics_type == "c2w":
                rotation, translation = _invert_extrinsics(rotation, translation)

            # Coordinate conversion (Unity-style camera Y-up -> OpenCV/GEN3C camera Y-down).
            #
            # IMPORTANT:
            # A plain left-multiply R' = S R with S=diag(1,-1,1) makes det(R') negative
            # (an improper rotation / reflection), which breaks quaternion-based interpolation
            # in _resample_camera_sequence() and can lead to black renders.
            #
            # Use a similarity transform in camera coordinates instead:
            #   R' = S R S,  t' = S t
            # This keeps det(R')=+1 for proper rotations.
            cam_basis = torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=device,
            )
            rotation = cam_basis @ rotation @ cam_basis
            translation = cam_basis @ translation

            w2c = torch.eye(4, dtype=torch.float32, device=device)
            w2c[:3, :3] = rotation
            w2c[:3, 3] = translation

            intrinsics = torch.tensor(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=device,
            )

            w2c_mats.append(w2c)
            intrinsic_mats.append(intrinsics)

    if not w2c_mats:
        raise ValueError(f"No camera entries parsed from {path}")

    w2c_tensor = torch.stack(w2c_mats, dim=0)
    intrinsics_tensor = torch.stack(intrinsic_mats, dim=0)

    if expected_frames is not None and expected_frames > 0:
        w2c_tensor, intrinsics_tensor = _resample_camera_sequence(
            w2c_tensor, intrinsics_tensor, expected_frames
        )

    return w2c_tensor.unsqueeze(0), intrinsics_tensor.unsqueeze(0)


def _matrix_to_quaternion(rotation: torch.Tensor) -> torch.Tensor:
    """Convert a 3x3 rotation matrix to a (w, x, y, z) quaternion."""
    m00, m01, m02 = rotation[0]
    m10, m11, m12 = rotation[1]
    m20, m21, m22 = rotation[2]

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = torch.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = torch.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = torch.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = torch.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    quat = torch.tensor([w, x, y, z], dtype=rotation.dtype, device=rotation.device)
    quat = quat / torch.linalg.norm(quat)
    return quat


def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert a (w, x, y, z) quaternion to a 3x3 rotation matrix."""
    q = quaternion / torch.linalg.norm(quaternion)
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return torch.tensor(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=quaternion.dtype,
        device=quaternion.device,
    )


def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    q0 = q0 / torch.linalg.norm(q0)
    q1 = q1 / torch.linalg.norm(q1)
    dot = torch.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / torch.linalg.norm(result)

    theta_0 = torch.acos(torch.clamp(dot, -1.0, 1.0))
    theta = theta_0 * t
    q2 = q1 - q0 * dot
    q2 = q2 / torch.linalg.norm(q2)
    return q0 * torch.cos(theta) + q2 * torch.sin(theta)


def _resample_camera_sequence(
    w2c_tensor: torch.Tensor, intrinsics_tensor: torch.Tensor, target_frames: int
) -> tuple[torch.Tensor, torch.Tensor]:
    original_frames = w2c_tensor.shape[0]
    if original_frames == target_frames:
        return w2c_tensor, intrinsics_tensor

    if original_frames == 1:
        w2c_resampled = w2c_tensor.repeat(target_frames, 1, 1)
        intrinsics_resampled = intrinsics_tensor.repeat(target_frames, 1, 1)
        return w2c_resampled, intrinsics_resampled

    device = w2c_tensor.device
    rotations = w2c_tensor[:, :3, :3]
    translations = w2c_tensor[:, :3, 3]
    intrinsics = intrinsics_tensor

    indices = torch.linspace(0, original_frames - 1, target_frames, device=device)
    idx0 = torch.floor(indices).long()
    idx1 = torch.clamp(idx0 + 1, max=original_frames - 1)
    alphas = indices - idx0.float()

    resampled_rot = []
    resampled_trans = []
    resampled_intrinsics = []

    for frame_idx in range(target_frames):
        lo = idx0[frame_idx].item()
        hi = idx1[frame_idx].item()
        alpha_t = float(alphas[frame_idx].item())
        if lo == hi or alpha_t == 0.0:
            resampled_rot.append(rotations[lo].clone())
            resampled_trans.append(translations[lo].clone())
            resampled_intrinsics.append(intrinsics[lo].clone())
            continue

        q0 = _matrix_to_quaternion(rotations[lo])
        q1 = _matrix_to_quaternion(rotations[hi])
        q_interp = _slerp(q0, q1, alpha_t)
        resampled_rot.append(_quaternion_to_matrix(q_interp))

        resampled_trans.append(((1 - alpha_t) * translations[lo] + alpha_t * translations[hi]).clone())
        resampled_intrinsics.append(((1 - alpha_t) * intrinsics[lo] + alpha_t * intrinsics[hi]).clone())

    w2c_final = torch.zeros((target_frames, 4, 4), dtype=w2c_tensor.dtype, device=device)
    w2c_final[:, :3, :3] = torch.stack(resampled_rot)
    w2c_final[:, :3, 3] = torch.stack(resampled_trans)
    w2c_final[:, 3, 3] = 1.0

    intrinsics_final = torch.stack(resampled_intrinsics)

    return w2c_final, intrinsics_final


def _align_inv_depth_to_depth(
    source_inv_depth: torch.Tensor,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply affine transformation to align source inverse depth to target depth.

    Args:
        source_inv_depth: Inverse depth map to be aligned. Shape: (H, W).
        target_depth: Target depth map. Shape: (H, W).
        target_mask: Mask of valid target pixels. Shape: (H, W).

    Returns:
        Aligned Depth map. Shape: (H, W).
    """
    target_inv_depth = 1.0 / target_depth
    source_mask = source_inv_depth > 0
    target_depth_mask = target_depth > 0

    if target_mask is None:
        target_mask = target_depth_mask
    else:
        target_mask = torch.logical_and(target_mask > 0, target_depth_mask)

    # Remove outliers
    outlier_quantiles = torch.tensor([0.1, 0.9], device=source_inv_depth.device)

    source_data_low, source_data_high = torch.quantile(source_inv_depth[source_mask], outlier_quantiles)
    target_data_low, target_data_high = torch.quantile(target_inv_depth[target_mask], outlier_quantiles)
    source_mask = (source_inv_depth > source_data_low) & (source_inv_depth < source_data_high)
    target_mask = (target_inv_depth > target_data_low) & (target_inv_depth < target_data_high)

    mask = torch.logical_and(source_mask, target_mask)

    source_data = source_inv_depth[mask].view(-1, 1)
    target_data = target_inv_depth[mask].view(-1, 1)

    ones = torch.ones((source_data.shape[0], 1), device=source_data.device)
    source_data_h = torch.cat([source_data, ones], dim=1)
    transform_matrix = torch.linalg.lstsq(source_data_h, target_data).solution

    scale, bias = transform_matrix[0, 0], transform_matrix[1, 0]
    aligned_inv_depth = source_inv_depth * scale + bias

    return 1.0 / aligned_inv_depth


def align_depth(
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor,
    k: torch.Tensor = None,
    c2w: torch.Tensor = None,
    alignment_method: str = "rigid",
    num_iters: int = 100,
    lambda_arap: float = 0.1,
    smoothing_kernel_size: int = 3,
) -> torch.Tensor:
    if alignment_method == "rigid":
        source_inv_depth = 1.0 / source_depth
        source_depth = _align_inv_depth_to_depth(source_inv_depth, target_depth, target_mask)
        return source_depth
    elif alignment_method == "non_rigid":
        if k is None or c2w is None:
            raise ValueError("Camera intrinsics (k) and camera-to-world matrix (c2w) are required for non-rigid alignment")
            
        source_inv_depth = 1.0 / source_depth
        source_depth = _align_inv_depth_to_depth(source_inv_depth, target_depth, target_mask)
        
        # Initialize scale map
        sc_map = torch.ones_like(source_depth).float().to(source_depth.device).requires_grad_(True)
        optimizer = torch.optim.Adam(params=[sc_map], lr=0.001)
        
        # Unproject target depth
        target_unprojected = unproject_points(
            target_depth.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
            c2w.unsqueeze(0),  # Add batch dimension
            k.unsqueeze(0),  # Add batch dimension
            is_depth=True,
            mask=target_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        ).squeeze(0)  # Remove batch dimension
        
        # Create smoothing kernel
        smoothing_kernel = torch.ones(
            (1, 1, smoothing_kernel_size, smoothing_kernel_size),
            device=source_depth.device
        ) / (smoothing_kernel_size**2)
        
        for _ in range(num_iters):
            # Unproject scaled source depth
            source_unprojected = unproject_points(
                (source_depth * sc_map).unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
                c2w.unsqueeze(0),  # Add batch dimension
                k.unsqueeze(0),  # Add batch dimension
                is_depth=True,
                mask=target_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            ).squeeze(0)  # Remove batch dimension
            
            # Data loss
            data_loss = torch.abs(source_unprojected[target_mask] - target_unprojected[target_mask]).mean()
            
            # Apply smoothing filter to sc_map
            sc_map_reshaped = sc_map.unsqueeze(0).unsqueeze(0)
            sc_map_smoothed = F.conv2d(
                sc_map_reshaped,
                smoothing_kernel,
                padding=smoothing_kernel_size // 2
            ).squeeze(0).squeeze(0)
            
            # ARAP loss
            arap_loss = torch.abs(sc_map_smoothed - sc_map).mean()
            
            # Total loss
            loss = data_loss + lambda_arap * arap_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        return source_depth * sc_map
    else:
        raise ValueError(f"Unsupported alignment method: {alignment_method}")
