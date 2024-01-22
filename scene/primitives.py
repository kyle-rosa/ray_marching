import torch
import torch.nn as nn


class SDFSphere(nn.Module):
    def __init__(self, radii):
        super().__init__()
        self.refractive_index = 4
        self.radius = nn.Parameter(torch.tensor(radii))

    def forward(self, query_positions):
        signed_distance = (
            query_positions
            .norm(dim=-1, p=2, keepdim=True)
            .sub(self.radius)
        )
        return signed_distance#, self.refractive_index

class SDFBox(nn.Module):
    def __init__(self, halfsides):
        super().__init__()
        self.refractive_index = 4
        self.halfsides = nn.Parameter(torch.tensor(halfsides))

    def forward(self, query_positions):
        q = query_positions.abs().sub(self.halfsides)
        q_max = q.max(dim=-1, keepdim=True).values
        signed_distance = (
            q.where(q > 0., 0.)
            .norm(dim=-1, p=2, keepdim=True)
            .add(q_max.where(q_max < 0., 0.))
        )
        return signed_distance#, self.refractive_index


class SDFPlane(nn.Module):
    def __init__(self):
        super().__init__()
        self.refractive_index = 4

    def forward(self, query_positions):
        return query_positions[..., [0]]#, self.refractive_index


class SDFLine(nn.Module):
    def __init__(self, starts, ends, radii):
        super().__init__()
        self.refractive_index = 4
        self.start = nn.Parameter(torch.tensor(starts))
        self.end = nn.Parameter(torch.tensor(ends))
        self.radius = nn.Parameter(torch.tensor(radii))

    def forward(self, query_positions):
        AB = self.end.sub(self.start)
        length2 = AB.pow(2).sum(dim=-1, keepdim=True)
        AB_div_length2 = AB.div(length2)
        AP = query_positions.sub(self.start)
        signed_distance = (
            AP.mul(AB_div_length2).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
            .mul(AB).sub(AP)
            .norm(dim=-1, p=2, keepdim=True).sub(self.radius)
        )
        return signed_distance#, self.refractive_index


class SDFDisk(nn.Module):
    def __init__(self, radius):
        super().__init__()
        self.refractive_index = 4
        self.radius = nn.Parameter(torch.tensor(radius))

    def forward(self, query_positions):
        r_dist = (
            query_positions[..., [1, 2]]
            .norm(dim=-1, p=2, keepdim=True)
            .subtract(self.radius)
        )
        signed_distance = (
            torch.cat([query_positions[..., [0]], r_dist.where(r_dist > 0., 0.)], dim=-1)
            .norm(dim=-1, keepdim=True)
        )
        return  (
            signed_distance#, self.refractive_index
        )


class SDFTorus(nn.Module):
    def __init__(self, radii1, radii2):
        super().__init__()
        self.radii1 = nn.Parameter(torch.tensor(radii1))
        self.radii2 = nn.Parameter(torch.tensor(radii2))

    def forward(self, query_positions):
        signed_distance = torch.cat(
            [
                query_positions[..., [0, 2]].norm(dim=-1, p=2, keepdim=True).sub(self.radii1),
                query_positions[..., [1]],
            ], dim=-1
        ).norm(dim=-1, p=2, keepdim=True).subtract(self.radii2)
        return (
            signed_distance#, self.refractive_index
        )
