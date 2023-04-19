import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import quaternion as Q

_default_device = torch.device('cpu')
_default_dtype = torch.float32

# def make_ndc_coordinates(
#     pixels_resolution: Tensor,
#     dtype: torch.dtype = torch.float32
# ):
#     device = pixels_resolution.device
#     (W, H) = pixels_resolution.unbind(-1)
#     return torch.stack(torch.meshgrid(
#         [
#             torch.arange(start=(1 - W), end=(1 + W), step=2, device=device, dtype=dtype).div(W),
#             torch.arange(start=(H - 1), end=-(H + 1), step=-2, device=device, dtype=dtype).div(H),
#         ],
#         indexing='xy'
#     ), dim=-1)


# def make_ndc_coordinates(
#     device: torch.device = _default_device,
#     dtype: torch.dtype = _default_dtype,
#     px_width: int = 800,
#     px_height: int = 600,
# ):
    
#     return torch.stack(torch.meshgrid(
#         [
#             torch.arange(start=(1 - px_width), end=(1 + px_width), step=2, device=device, dtype=dtype).div(px_width),
#             torch.arange(start=(px_height - 1), end=-(px_height + 1), step=-2, device=device, dtype=dtype).div(px_height),
#         ],
#         indexing='xy'
#     ), dim=-1)


class PinholeCamera(nn.Module):
    def __init__(
        self,
        num_cameras: int,
        px_width: int,
        px_height: int,
        focal_length: float,
        sensor_width: float,
        sensor_height: float,
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.size = (num_cameras, 1, px_height, px_width)

        self.register_buffer('focus', torch.tensor([[[[0., 0., focal_length]]]]).expand(num_cameras, 1, 1, 3))
        self.register_buffer('theta', torch.tensor([[[sensor_height / 2, 0., 0.], [0., sensor_width / 2, 0.]]]).expand(num_cameras, 2, 3))
        self.register_buffer('ray_positions', 
            F.pad(F.affine_grid(theta=self.theta, size=self.size, align_corners=False), pad=[0, 1], value=0.)
        )
        self.register_buffer('ray_directions', F.normalize(self.focus.sub(self.ray_positions), p=2, dim=-1, eps=0))
        self.register_buffer('pixel_frames', torch.eye(3)[None, None, None, [0, 1], :].expand(num_cameras, 1, 1, 2, 3))

        self.quaternion_to_so3 = Q.QuaternionToSO3()

    def forward(self, orientation, translation):
        ray_positions = Q.rotation(self.ray_positions, orientation).add(translation)
        ray_directions = Q.rotation(self.ray_directions, orientation)
        pixel_frames = self.quaternion_to_so3(orientation)
        return (ray_positions, pixel_frames, ray_positions, ray_directions)


class Marcher(nn.Module):
    def __init__(
        self,
        sdf: nn.Module,
        marching_steps: int = 32
    ):
        super().__init__()
        self.sdf = sdf
        self.marching_steps = marching_steps

    def forward(self, ray_positions, ray_directions):
        for _ in range(self.marching_steps):
            ray_positions = self.sdf(ray_positions).mul(ray_directions).add(ray_positions)
        return ray_positions