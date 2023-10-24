import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import quaternion as Q


# Lambertian Shader
class LambertianShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ray_directions, surface_normals):
        return (
            ray_directions.mul(surface_normals)
            .sum(dim=-1).neg().clamp(0, 1)[..., None]
        )


# Distance Shader
class DistanceShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, px_coords, surface_coords, decay_factor):
        log_dists = (
            px_coords.sub(surface_coords)
            .norm(dim=-1, p=2, keepdim=True)
            .log()
        )
        return log_dists.sub(log_dists.min()).div(log_dists.max() - log_dists.min()).pow(1/2.33)


# Proximity Shader
class ProximityShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, surface_distances):
        log_dists = surface_distances.clamp(1e-3, float('inf')).log()
        return log_dists.sub(log_dists.min()).div(log_dists.max() - log_dists.min()).pow(1/2.33)


# Vignette Shader
class VignetteShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ray_directions, pixel_frames):
        return (
            ray_directions.mul(pixel_frames[..., 2])
            .sum(dim=-1, keepdim=True).pow(3)
        )


# Normals Shader
class NormalShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, surface_normals):
        return (
            surface_normals
            .add(1).div(2)
            .clamp(0, 1)
            .pow(1/2.33)
        )


# Laplacian Shader
class LaplacianShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, surface_laplacian):
        return (
            surface_laplacian
            .div(surface_laplacian.abs().max())
            .mul(-1)
            .add(1).div(2)
            .clamp(0, 1)
            .pow(1 / 2.33)
        )


# Tangents Shader
def angle_colouring(real_part, imag_part, cyclic_colourmap, degree):
    cmap_index = (
        torch.atan2(imag_part, real_part)
        .div(math.tau).add(0.5).mul(degree)
        .multiply(cyclic_colourmap.shape[0])
        .floor().long().remainder(cyclic_colourmap.shape[0])
    )
    return cyclic_colourmap[cmap_index, :]

def domain_colouring(real_part, imag_part, cyclic_colourmap, degree=1):
    return (
        angle_colouring(real_part, imag_part, cyclic_colourmap, degree)
        .multiply((real_part.pow(2).add(imag_part.pow(2))).pow(0.5)[..., None])
    )

class TangentShader(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(
            self,
            camera_orientation_conj,
            ray_directions,
            surface_normals,
            cyclic_colourmap,
            degree
        ):
        projected_normals = Q.rotation(
            (
                surface_normals
                .multiply(ray_directions)
                .sum(dim=-1, keepdim=True)
                .multiply(ray_directions)
                .neg()
                .add(surface_normals)
            ), 
            camera_orientation_conj
        )
        real_part = projected_normals[..., 0] 
        imag_part = projected_normals[..., 1]
        return domain_colouring(real_part, imag_part, cyclic_colourmap, degree)


# Spin Shader
class SpinShader(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(
            self,
            camera_orientation_conj,
            surface_normals,
            cyclic_colourmap,
            degree
        ):
        value = Q.multiply(
            F.pad(surface_normals, [1, 0], value=0.),
            camera_orientation_conj,
        )
        (a, bcd) = (value[..., 0], value[..., 1:])
        real_part = a.pow(2).subtract(bcd.pow(2).sum(dim=-1))
        imag_part = bcd.norm(dim=-1, p=2).mul(a).mul(2)
        return domain_colouring(real_part, imag_part, cyclic_colourmap, degree)


# All Shaders
class Shader(nn.Module):
    def __init__(
        self,
        cyclic_cmap: Tensor = torch.load(Path() / 'data/cyclic_cmap.pt'),
        decay_factor: float = 0.01,
        dtype=torch.float,
    ):
        super().__init__()
        self.register_buffer('cyclic_cmap', cyclic_cmap.clone().to(dtype))
        self.register_buffer('decay_factor', torch.tensor(decay_factor, dtype=dtype))

        self.lambertian_shader = LambertianShader()
        self.normal_shader = NormalShader()
        self.tangent_shader = TangentShader()
        self.spin_shader = SpinShader()
        self.distance_shader = DistanceShader()
        self.proximity_shader = ProximityShader()
        self.vignette_shader = VignetteShader()
        self.laplacian_layer = LaplacianShader()

    def forward(
        self, 
        px_coords: Tensor,
        camera_orientation: Tensor,
        pixel_frames: Tensor,
        ray_directions: Tensor,
        surface_coords: Tensor,
        surface_normals: Tensor,
        surface_laplacian: Tensor,
        surface_distances: Tensor,
        degree: int,
    ):
        self.cyclic_cmap = self.cyclic_cmap.roll(65, -2)
        camera_orientation_conj = Q.conjugate(
            camera_orientation
        )[..., None, None, :]
        print(camera_orientation_conj.shape)
        lambertian_layer = self.lambertian_shader(
            ray_directions,
            surface_normals
        )
        distance_layer = self.distance_shader(
            px_coords,
            surface_coords,
            self.decay_factor
        )
        proximity_layer = self.proximity_shader(
            surface_distances
        )
        vignette_layer = self.vignette_shader(
            ray_directions,
            pixel_frames
        )
        normal_layer = self.normal_shader(
            surface_normals
        )
        laplacian_layer = self.laplacian_layer(
            surface_laplacian
        )
        tangent_layer = self.tangent_shader(
            camera_orientation_conj,
            ray_directions,
            surface_normals,
            self.cyclic_cmap,
            degree
        )
        spin_layer = self.spin_shader(
            camera_orientation_conj,
            surface_normals,
            self.cyclic_cmap,
            degree
        )

        return (
            lambertian_layer,
            distance_layer,
            proximity_layer,
            vignette_layer,
            normal_layer,
            laplacian_layer,
            tangent_layer,
            spin_layer
        )


def aggregate_rays(
        px_width,
        px_height,
        points_screen,
        ray_features,
    ):
        shape = (px_height, px_width, ray_features.shape[-1])
        points_screen_idx = torch.stack(
            [ 
                points_screen[..., 0].add(1).div(2).mul(px_width).trunc().long().clamp(0, px_width-1),
                points_screen[..., 1].mul(-1).add(1).div(2).mul(px_height).trunc().long().clamp(0, px_height-1)
            ], dim=-1
        )
        points_screen_idx_linear = (
            points_screen_idx[..., 1] * px_width
            + points_screen_idx[..., 0]
        )
        numer = (
            torch.zeros(shape, dtype=ray_features.dtype, device=ray_features.device)
            .view((px_height * px_width), ray_features.shape[-1])
            .index_add(dim=0, index=points_screen_idx_linear, source=ray_features)
        )
        denom = (
            torch.zeros(shape, dtype=ray_features.dtype, device=ray_features.device)
            .view(px_height * px_width, ray_features.shape[-1])
            .index_add(dim=0, index=points_screen_idx_linear, source=torch.ones_like(ray_features))
        )
        return numer.div(denom).where(denom!=0, 0.).view(shape)
