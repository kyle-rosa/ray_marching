import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import quaternion as Q


class LambertianShader(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, ray_directions: Tensor, surface_normals: Tensor) -> Tensor:
        return (
            ray_directions.mul(surface_normals)
            .sum(dim=-1, keepdim=True).neg().clamp(0, 1)
        )


class DistanceShader(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, px_coords: Tensor, surface_coords: Tensor) -> Tensor:
        log_dists = (
            px_coords.sub(surface_coords)
            .norm(dim=-1, p=2, keepdim=True)
            .clamp(1e-2, float('inf'))
            .log()
        )
        return (
            log_dists.sub(log_dists.min())
            .div(log_dists.max().sub(log_dists.min()))
            .pow(1 / 2.33)
        )


class ProximityShader(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, surface_distances: Tensor) -> Tensor:
        log_dists = (
            surface_distances
            .clamp(1e-2, float('inf'))
            .log()
        )
        return (
            log_dists.sub(log_dists.min())
            .div(log_dists.max().sub(log_dists.min()))
            .pow(1 / 2.33)
        )


class VignetteShader(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, ray_directions: Tensor, pixel_frames: Tensor) -> Tensor:
        return (
            ray_directions.mul(pixel_frames[..., 2])
            .sum(dim=-1, keepdim=True).pow(3)
        )


class NormalShader(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, surface_normals: Tensor) -> Tensor:
        return surface_normals.abs().clamp(0, 1)


class LaplacianShader(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, surface_laplacian: Tensor) -> Tensor:
        return (
            surface_laplacian
            .div(surface_laplacian.abs().max())
            .mul(-1)
            .add(1).div(2)
            .clamp(0, 1)
            .pow(1 / 2.33)
        )


def angle_colouring(
    real_part: Tensor,
    imag_part: Tensor,
    cyclic_colourmap: Tensor,
    degree: int
) -> Tensor:
    cmap_index = (
        torch.atan2(imag_part, real_part)
        .div(math.tau).add(0.5).mul(degree)
        .multiply(cyclic_colourmap.shape[0])
        .floor().long().remainder(cyclic_colourmap.shape[0])
    )
    return cyclic_colourmap[cmap_index, :]


def domain_colouring(
    real_part: torch.Tensor,
    imag_part: torch.Tensor,
    cyclic_colourmap: torch.Tensor,
    degree: int
) -> Tensor:
    colours = angle_colouring(real_part, imag_part, cyclic_colourmap, degree)
    brightness = (
        torch.stack([real_part, imag_part], dim=-1)
        .pow(2).sum(dim=-1, keepdim=True).pow(1 / 2)
    )
    return brightness.mul(colours)


class TangentShader(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        camera_orientation_conj: Tensor,
        ray_directions: Tensor,
        surface_normals: Tensor,
        cyclic_colourmap: Tensor,
        degree: int = 1
    ) -> Tensor:
        projected_normals = Q.rotation(
            (
                surface_normals
                .multiply(ray_directions)
                .sum(dim=-1, keepdim=True)
                .multiply(ray_directions)
                .mul(-1.0)
                .add(surface_normals)
            ),
            camera_orientation_conj
        )
        tangent_image = domain_colouring(
            projected_normals[..., 0],
            projected_normals[..., 1],
            cyclic_colourmap,
            degree
        )
        return tangent_image


class SpinShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        camera_orientation_conj: Tensor,
        surface_normals: Tensor,
        cyclic_colourmap: Tensor,
        degree: int = 1
    ) -> Tensor:
        value = Q.multiply(
            F.pad(surface_normals, [1, 0], value=0.0),
            camera_orientation_conj,
        )
        (a, bcd) = (value[..., 0], value[..., 1:])
        real_part = a.pow(2).subtract(bcd.pow(2).sum(dim=-1))
        imag_part = bcd.norm(dim=-1, p=2).mul(a).mul(2)
        return domain_colouring(imag_part, real_part, cyclic_colourmap, degree)


class Shader(nn.Module):
    def __init__(self):
        super().__init__()
        cyclic_cmap = torch.load(Path('./data/cyclic_cmap.pt'), weights_only=True)
        self.register_buffer("cyclic_cmap", cyclic_cmap)
        print(f"{self.cyclic_cmap.dtype=}, {self.cyclic_cmap.shape=}")

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
        mode: int,
        degree: int,
    ) -> Tensor:
        # self.cyclic_cmap = self.cyclic_cmap.roll(65, -2)
        modes = [
            'lambertian', 'distance', 'proximity',
            'vignette', 'normal', 'laplacian',
            'tangent', 'spin'
        ]
        mode = modes[mode % len(modes)]
        if mode == "lambertian":
            image = self.lambertian_shader(
                ray_directions,
                surface_normals
            )
        elif mode == "distance":
            image = self.distance_shader(
                px_coords,
                surface_coords,
            )
        elif mode == "proximity":
            image = self.proximity_shader(
                surface_distances
            )
        elif mode == "vignette":
            image = self.vignette_shader(
                ray_directions,
                pixel_frames
            )
        elif mode == "normal":
            image = self.normal_shader(
                surface_normals
            )
        elif mode == "laplacian":
            image = self.laplacian_layer(
                surface_laplacian
            )

        elif mode == "tangent":
            camera_orientation_conj = Q.conjugate(
                camera_orientation
            )[:, None, None, :]
            image = self.tangent_shader(
                camera_orientation_conj,
                ray_directions,
                surface_normals,
                self.cyclic_cmap,
                degree
            )
        elif mode == "spin":
            camera_orientation_conj = Q.conjugate(
                camera_orientation
            )[:, None, None, :]
            image = self.spin_shader(
                camera_orientation_conj,
                surface_normals,
                self.cyclic_cmap,
                degree
            )
        else:
            print(f"{mode=} rendering mode not implemented.")
            raise NotImplementedError()

        return image


class OmniShader(nn.Module):
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
