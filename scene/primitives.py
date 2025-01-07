import torch
import torch.nn as nn
from torch import Tensor


class SDFSphere(nn.Module):
    def __init__(self, radius: float):
        super().__init__()
        self.radius = nn.Parameter(torch.tensor(radius))

    def forward(self, query_positions: Tensor) -> Tensor:
        signed_distance = (
            query_positions
            .norm(dim=-1, p=2, keepdim=True)
            .sub(self.radius)
        )
        return signed_distance


class SDFBox(nn.Module):
    def __init__(self, halfsides: tuple[float, float, float]):
        super().__init__()
        self.halfsides = nn.Parameter(torch.tensor(halfsides))

    def forward(self, query_positions: Tensor) -> Tensor:
        q = query_positions.abs().sub(self.halfsides)
        q_max = q.max(dim=-1, keepdim=True).values
        signed_distance = (
            q.where(q > 0., 0.)
            .norm(dim=-1, p=2, keepdim=True)
            .add(q_max.where(q_max < 0., 0.))
        )
        return signed_distance


class SDFPlane(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_positions: Tensor) -> Tensor:
        return query_positions[..., [0]]


class SDFLine(nn.Module):
    def __init__(self, start: float, end: float, radius: float):
        super().__init__()
        self.start = nn.Parameter(torch.tensor(start))
        self.end = nn.Parameter(torch.tensor(end))
        self.radius = nn.Parameter(torch.tensor(radius))

    def forward(self, query_positions: Tensor) -> Tensor:
        AB = self.end.sub(self.start)
        length2 = AB.pow(2).sum(dim=-1, keepdim=True)
        AB_div_length2 = AB.div(length2)
        AP = query_positions.sub(self.start)
        signed_distance = (
            AP.mul(AB_div_length2).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
            .mul(AB).sub(AP)
            .norm(dim=-1, p=2, keepdim=True).sub(self.radius)
        )
        return signed_distance


class SDFDisk(nn.Module):
    def __init__(self, radius: float):
        super().__init__()
        self.radius = nn.Parameter(torch.tensor(radius))

    def forward(self, query_positions: Tensor) -> Tensor:
        r_dist = (
            query_positions[..., [1, 2]]
            .norm(dim=-1, p=2, keepdim=True)
            .subtract(self.radius)
        )
        signed_distance = (
            torch.cat(
                [query_positions[..., [0]], r_dist.where(r_dist > 0., 0.)],
                dim=-1
            )
            .norm(dim=-1, keepdim=True)
        )
        return signed_distance


class SDFTorus(nn.Module):
    def __init__(self, radius1: float, radius2: float) -> None:
        super().__init__()
        self.radius1 = nn.Parameter(torch.tensor(radius1))
        self.radius2 = nn.Parameter(torch.tensor(radius2))

    def forward(self, query_positions: Tensor) -> Tensor:
        signed_distance = torch.cat(
            [
                (
                    query_positions[..., [0, 2]]
                    .norm(dim=-1, p=2, keepdim=True)
                    .sub(self.radius1)
                ),
                query_positions[..., [1]],
            ], dim=-1
        ).norm(dim=-1, p=2, keepdim=True).subtract(self.radius2)
        return signed_distance
