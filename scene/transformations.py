import torch
import torch.nn as nn

import quaternion as Q


class SDFAffineTransformation(nn.Module):
    def __init__(self, sdf, orientation, translation):
        super().__init__()
        self.sdf = sdf
        self.register_buffer('orientation', torch.tensor(orientation))
        self.register_buffer('translation', torch.tensor(translation))
        self.register_buffer('orientation_conj',  Q.conjugate(self.orientation))

    def forward(self, query_positions):
        return self.sdf(
            Q.rotation(
                query_positions[..., None, :].sub(self.translation), 
                self.orientation_conj
            )
        )


class SDFSmoothUnion(nn.Module):
    def __init__(self, sdfs, blend_k):
        super().__init__()
        self.sdfs = nn.ModuleList(sdfs)
        self.register_buffer('blend_k', torch.tensor(blend_k))

    def forward(self, query_coords):
        return (
            torch.cat([sdf(query_coords) for sdf in self.sdfs], dim=-2)
            .mul(-self.blend_k).logsumexp(dim=-2).div(-self.blend_k)
        )


class SDFUnion(nn.Module):
    def __init__(self, sdfs):
        super().__init__()
        self.sdfs = nn.ModuleList(sdfs)

    def forward(self, query_coords):
        return (
            torch.cat([sdf(query_coords) for sdf in self.sdfs], dim=-2)
            .min(dim=-2).values
        )

    
class SDFRounding(nn.Module):
    def __init__(self, sdf, rounding):
        super().__init__()
        self.sdf = sdf
        self.register_buffer('rounding', torch.tensor(rounding))

    def forward(self, query_coords):
        return self.sdf(query_coords).subtract(self.rounding)


class SDFOnion(nn.Module):
    def __init__(self, sdf, radii):
        super().__init__()
        self.sdf = sdf
        self.register_buffer('radii',  torch.tensor(radii))

    def forward(self, query_coords):
        return self.sdf(query_coords).abs().sub(self.radii)
