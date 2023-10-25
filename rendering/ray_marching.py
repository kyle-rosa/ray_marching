import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import quaternion as Q


_default_dtype = torch.float32


class PinholeCamera(nn.Module):
    def __init__(
        self,
        num_cameras: int,
        px_width: int,
        px_height: int,
        focal_length: float,
        sensor_width: float,
        sensor_height: float,
        dtype: torch.dtype = _default_dtype
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
        pixel_frames = self.quaternion_to_so3(orientation[:, None, None, :])
        return (ray_positions, pixel_frames, ray_positions, ray_directions)


class RayGenerator(nn.Module):
    def __init__(
            self,
            focal_length = 17e-3,
            sensor_width = 17e-3,
            sensor_height = 17e-3
        ):
        super().__init__()
        self.sensor_width = nn.Parameter(torch.tensor(sensor_width))
        self.sensor_height = nn.Parameter(torch.tensor(sensor_height))
        self.focal_length = nn.Parameter(torch.tensor(focal_length))
        self.quaternion_to_so3 = Q.QuaternionToSO3()
    
    def forward(self, points_screen, orientation: Tensor, translation: Tensor) -> Tensor:
        points_space = torch.stack(
            [
                points_screen[..., 0] * self.sensor_width,
                points_screen[..., 1] * self.sensor_height,
                self.focal_length.expand_as(points_screen[..., 0])
            ], dim=-1
        )
        ray_directions = torch.nn.functional.normalize(points_space, dim=-1, p=2)

        ray_positions = Q.rotation(points_space, orientation).add(translation)
        ray_directions = Q.rotation(ray_directions, orientation)
        pixel_frames = self.quaternion_to_so3(orientation[:, None, :])
        return (ray_positions, pixel_frames, ray_positions, ray_directions)


# def reflection_directions(surface_normals, ray_directions):



class SDFMarcher(nn.Module):
    def __init__(
        self,
        sdf_scene: nn.Module,
        marching_steps: int = 32
    ):
        super().__init__()
        self.sdf_scene = sdf_scene
        self.marching_steps = marching_steps

    def forward(self, ray_positions: Tensor, ray_directions: Tensor, marching_steps: int = 32) -> Tensor:
        for _ in range(marching_steps):
            # print(dists.shape)
            ray_positions = self.sdf_scene(ray_positions).mul(ray_directions).add(ray_positions)
        return ray_positions


class SDFNormals(nn.Module):
    def __init__(
            self, 
            sdf_scene: nn.Module,
            normals_eps: float = 1e-3,
            dtype: torch.dtype = _default_dtype
        ):
        super().__init__()
        self.normals_eps = normals_eps
        self.register_buffer(
            'offsets',
            torch.tensor(
                [
                    [1., 0., -0.5**0.5],
                    [-1., 0., -0.5**0.5],
                    [0., 1., 0.5**0.5],
                    [0., -1., 0.5**0.5],
                ], 
                dtype=torch.double
            )
        )
        self.offsets = F.normalize(self.offsets, dim=-1, p=2, eps=0.)
        self.offsets = self.offsets.mul(self.normals_eps)
        self.register_buffer(
            'relative_offsets', 
            self.offsets[..., [1, 2, 3], :].sub(self.offsets[..., [0], :])
        )
        self.register_buffer('offsets_inverse', self.relative_offsets.inverse())
        self.offsets = self.offsets.to(dtype)
        self.relative_offsets = self.relative_offsets.to(dtype)
        self.offsets_inverse = self.offsets_inverse.to(dtype)

        self.sdf_scene = sdf_scene
        
    def forward(self, surface_coords):
        offset_values = self.sdf_scene(surface_coords[..., None, :].add(self.offsets))
        d_values = offset_values[..., [1, 2, 3], :].sub(offset_values[..., [0], :])
        normals = self.offsets_inverse.mul(d_values[..., None, :, 0]).sum(dim=-1)
        normals = F.normalize(normals, dim=-1, p=2, eps=0.)
        laplacian = (
            self.sdf_scene(surface_coords)
            .sub(offset_values.mean(dim=-2))
            .mul(6 / self.normals_eps**2)
        )
        return (normals, laplacian)
