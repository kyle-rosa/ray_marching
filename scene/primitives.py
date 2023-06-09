import torch
import torch.nn as nn


class SDFSpheres(nn.Module):
    def __init__(self, radii):
        super().__init__()
        self.register_buffer('radii',  torch.tensor(radii))

    def forward(self, query_positions):
        return (
            query_positions
            .norm(dim=-1, p=2, keepdim=True)
            .sub(self.radii)
        )


class SDFBoxes(nn.Module):
    def __init__(self, halfsides):
        super().__init__()
        self.register_buffer('halfsides',  torch.tensor(halfsides))

    def forward(self, query_positions):
        q = query_positions.abs().sub(self.halfsides)
        q_max = q.max(dim=-1, keepdim=True).values
        return (
            q.where(q > 0., 0.)
            .norm(dim=-1, p=2, keepdim=True)
            .add(q_max.where(q_max < 0., 0.))
        )


class SDFPlanes(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, query_positions):
        return query_positions[..., [0]]


class SDFLine(nn.Module):
    def __init__(self, starts, ends, radii):
        super().__init__()
        self.register_buffer('starts', torch.tensor(starts))
        self.register_buffer('ends', torch.tensor(ends))
        self.register_buffer('radii', torch.tensor(radii))

        self.register_buffer('AB', self.ends.sub(self.starts))
        self.register_buffer('length2', self.AB.pow(2).sum(dim=-1, keepdim=True))
        self.register_buffer('AB_div_length2', self.AB.div(self.length2))

    def forward(self, query_positions):
        AP = query_positions[..., None, :].sub(self.starts)
        return (
            AP.mul(self.AB_div_length2).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
            .mul(self.AB).sub(AP)
            .norm(dim=-1, p=2, keepdim=True).sub(self.radii)
        )


class SDFDisks(nn.Module):
    def __init__(self, radii):
        super().__init__()
        self.register_buffer('radii',  torch.tensor(radii))

    def forward(self, query_positions):
        r_dist = (
            query_positions[..., [1, 2]]
            .norm(dim=-1, p=2, keepdim=True)
            .subtract(self.radii)
        )
        return  (
            torch.cat([query_positions[..., [0]], r_dist.where(r_dist > 0., 0.)], dim=-1)
            .norm(dim=-1, keepdim=True)
        )


class SDFTorii(nn.Module):
    def __init__(self, radii1, radii2):
        super().__init__()
        self.register_buffer('radii1',  torch.tensor(radii1))
        self.register_buffer('radii2',  torch.tensor(radii2))

    def forward(self, query_positions):
        return torch.cat(
            [
                query_positions[..., [0, 2]].norm(dim=-1, p=2, keepdim=True).sub(self.radii1),
                query_positions[..., [1]],
            ], dim=-1
        ).norm(dim=-1, p=2, keepdim=True).subtract(self.radii2)
