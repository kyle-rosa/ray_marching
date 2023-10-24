import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import quaternion as Q


# Lambertian Shader
def lambertian_shader(ray_directions, surface_normals):
    return ray_directions.mul(surface_normals).sum(dim=-1).neg().clamp(0, 1)[..., None]

class LambertianShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ray_directions, surface_normals):
        return lambertian_shader(ray_directions, surface_normals)


# Distance Shader
def distance_shader(px_coords, surface_coords, decay_factor):
    return px_coords.sub(surface_coords).norm(dim=-1, p=2, keepdim=True).mul(-decay_factor).exp()

class DistanceShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, px_coords, surface_coords, decay_factor):
        return distance_shader(px_coords, surface_coords, decay_factor)


# Proximity Shader
def proximity_shader(surface_distances):
    return surface_distances.mul(-300).sigmoid()

class ProximityShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, surface_distances):
        return proximity_shader(surface_distances)


# Vignette Shader
def vignette_shader(ray_directions, px_normals):
    return ray_directions.mul(px_normals).sum(dim=-1, keepdim=True).clamp(0, 1)

class VignetteShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ray_directions, pixel_frames):
        px_normals = pixel_frames[..., 2]
        return vignette_shader(ray_directions, px_normals)


# Normals Shader
def normal_shader(surface_normals):
    return surface_normals.abs()

class NormalShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, surface_normals):
        return normal_shader(surface_normals)


# Laplacian Shader
def laplacian_shader(surface_laplacian):
    return surface_laplacian.div(surface_laplacian.abs().max()).mul(-1).add(1).div(2)

class LaplacianShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, surface_laplacian):
        return laplacian_shader(surface_laplacian)


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

def project_normals(camera_orientation_conj, surface_normals, ray_directions):
    return Q.rotation(
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

def tangents_shader(camera_orientation_conj, ray_directions, surface_normals, cyclic_colourmap, degree):
    projected_normals = project_normals(camera_orientation_conj, surface_normals, ray_directions)
    return domain_colouring(projected_normals[..., 0], projected_normals[..., 1], cyclic_colourmap, degree)

class TangentShader(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, camera_orientation_conj, ray_directions, surface_normals, cyclic_colourmap, degree):
        return tangents_shader(camera_orientation_conj, ray_directions, surface_normals, cyclic_colourmap, degree)


# Spin Shader
def spin_shader(camera_orientation_conj, surface_normals, cyclic_colourmap, degree):
    value = Q.multiply(
        F.pad(surface_normals, [1, 0], value=0.),
        camera_orientation_conj,
    )
    (a, bcd) = (value[..., 0], value[..., 1:])
    real_part = a.pow(2).subtract(bcd.pow(2).sum(dim=-1))
    imag_part = bcd.norm(dim=-1, p=2).mul(a).mul(2)
    return domain_colouring(real_part, imag_part, cyclic_colourmap, degree)


class SpinShader(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, camera_orientation_conj, surface_normals, cyclic_colourmap, degree):
        return spin_shader(camera_orientation_conj, surface_normals, cyclic_colourmap, degree)


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
