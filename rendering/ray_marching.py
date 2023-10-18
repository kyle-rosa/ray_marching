import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import quaternion as Q


class PinholeCamera(nn.Module):
    def __init__(
        self,
        num_cameras: int,
        px_width: int,
        px_height: int,
        focal_length: float,
        sensor_width: float,
        sensor_height: float,
        dtype=torch.float32
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.size = (num_cameras, 1, px_height, px_width)

        self.register_buffer(
            'focus',
            torch.tensor([[[[0., 0., -focal_length]]]], dtype=dtype).expand(num_cameras, 1, 1, 3)
        )
        self.register_buffer(
            'theta',
            torch.tensor([[[sensor_width / 2, 0., 0.], [0., -sensor_height / 2, 0.]]], dtype=dtype).expand(num_cameras, 2, 3)
        )
        self.register_buffer(
            'ray_positions',
            F.pad(F.affine_grid(theta=self.theta.float(), size=self.size, align_corners=False), pad=[0, 1], value=0.).to(dtype=dtype)
        )
        self.register_buffer(
            'ray_directions',
            F.normalize(self.ray_positions.sub(self.focus).cuda(), p=2, dim=-1, eps=0)
        )
        self.register_buffer(
            'pixel_frames',
            torch.eye(3, dtype=dtype)[None, None, None, [0, 1], :].expand(num_cameras, 1, 1, 2, 3)
        )
        self.quaternion_to_so3 = Q.QuaternionToSO3()

    def forward(self, orientation: Tensor, translation: Tensor) -> Tensor:
        ray_positions = Q.rotation(self.ray_positions, orientation).add(translation)
        ray_directions = Q.rotation(self.ray_directions, orientation)
        pixel_frames = self.quaternion_to_so3(orientation)
        return (ray_positions, pixel_frames, ray_positions, ray_directions)


class Marcher(nn.Module):
    def __init__(
        self,
        sdf_scene: nn.Module,
        marching_steps: int = 32
    ):
        super().__init__()
        self.sdf_scene = sdf_scene
        self.marching_steps = marching_steps

    def forward(self, ray_positions: Tensor, ray_directions: Tensor) -> Tensor:
        for _ in range(self.marching_steps):
            ray_positions = self.sdf_scene(ray_positions).mul(ray_directions).add(ray_positions)
        return ray_positions