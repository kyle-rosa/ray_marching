import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import quaternion as Q


class SDFNormals(torch.nn.Module):
    def __init__(self, sdf_scene):
        super().__init__()
        self.register_buffer(
            'offsets',
            torch.tensor(
                [
                    [1., 0., -0.5**0.5],
                    [-1., 0., -0.5**0.5],
                    [0., 1., 0.5**0.5],
                    [0., -1., 0.5**0.5],
                ]
            ).mul(1e-4)
        )
        self.register_buffer(
            'relative_offsets', 
            self.offsets[..., [1, 2, 3], :].sub(self.offsets[..., [0], :])
        )
        self.register_buffer('offsets_inverse', self.relative_offsets.inverse())
        self.sdf_scene = sdf_scene
        
    def forward(self, surface_coords):
        offset_values = self.sdf_scene(surface_coords[..., None, :].add(self.offsets))
        d_values = offset_values[..., [1, 2, 3], :].sub(offset_values[..., [0], :])
        return F.normalize(
            self.offsets_inverse.mul(d_values[..., None, :, 0]).sum(dim=-1),
            dim=-1, p=2, eps=0.
        )


def functional_make_sdf_distance_and_normal(sdf_scene, surface_coords):
    (distances, vjp_fn) = torch.func.vjp(sdf_scene, surface_coords, has_aux=False)
    surface_normals = vjp_fn(torch.ones_like(distances))[-1]
    return (distances, surface_normals)


def lambertian_shader(ray_directions, surface_normals):
    return ray_directions.mul(surface_normals).sum(dim=-1).neg().clamp(0, 1)[..., None]


class LambertianShader(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ray_directions, surface_normals):
        return lambertian_shader(ray_directions, surface_normals)


def distance_shader(px_coords, surface_coords, decay_factor):
    return px_coords.sub(surface_coords).norm(dim=-1, p=2, keepdim=True).mul(-decay_factor).exp()


def proximity_shader(surface_distances):
    return surface_distances.mul(-300).sigmoid()


def vignette_shader(ray_directions, px_normals):
    return ray_directions.mul(px_normals).sum(dim=-1, keepdim=True).clamp(0, 1)


def normal_shader(surface_normals):
    return surface_normals.abs()


class NormalShader(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, surface_normals):
        return normal_shader(surface_normals)


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


class TangentShader(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, camera_orientation_conj, ray_directions, surface_normals, cyclic_colourmap, degree):
        return tangents_shader(camera_orientation_conj, ray_directions, surface_normals, cyclic_colourmap, degree)


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


class Shader(nn.Module):
    def __init__(
        self,
        cyclic_cmap: Tensor = torch.load(Path() / 'data/cyclic_cmap.pt'),
        decay_factor: float = 0.01
    ):
        super().__init__()
        self.register_buffer('cyclic_cmap', cyclic_cmap.clone())
        self.register_buffer('decay_factor', torch.tensor(decay_factor))

        self.lambertian_shader = LambertianShader()
        self.normal_shader = NormalShader()
        self.tangent_shader = TangentShader()
        self.spin_shader = SpinShader()

    def forward(
        self, 
        px_coords: Tensor, 
        camera_orientation: Tensor,
        px_normals: Tensor, 
        ray_directions: Tensor, 
        surface_coords: Tensor,
        surface_normals: Tensor, 
        degree: int,
    ):
        self.cyclic_cmap = self.cyclic_cmap.roll(65, -2)
        camera_orientation_conj = Q.conjugate(camera_orientation)[..., None, None, :]
        lambertian_layer = self.lambertian_shader(
            ray_directions,
            surface_normals
        )
        normal_layer = self.normal_shader(
            surface_normals
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
        base_layer = torch.ones_like(normal_layer)
        return (base_layer, (lambertian_layer, normal_layer, tangent_layer, spin_layer))
