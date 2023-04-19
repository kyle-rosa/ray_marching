import torch
import torch.nn as nn
from torch import Tensor

import quaternion as Q


class SDFAffineTransformation(nn.Module):
    """
    Given an SDF and an affine transformation, returns a new SDF for the transformed geometry.
    This is done by applying the inverse transformation to the queries, then evaluating the original SDF.

    Args:
        sdf (nn.Module): A callable that takes a 3D tensor of query positions as input and
                         returns a tensor of signed distances.
        orientation (list, tuple or numpy.ndarray): A quaternion that specifies the rotation of the
                                                    input positions.
        translation (list, tuple or numpy.ndarray): A 3-element vector that specifies the translation
                                                    of the input positions.
    """
    def __init__(
        self,
        sdf: nn.Module,
        orientation: tuple[float, float, float, float],
        translation: tuple[float, float, float]
    ):
        super().__init__()
        self.sdf = sdf
        self.register_buffer('orientation', torch.tensor(orientation))
        self.register_buffer('translation', torch.tensor(translation))
        self.register_buffer('orientation_conj',  Q.conjugate(self.orientation))

    def forward(
        self, 
        query_positions: Tensor
    ) -> Tensor:
        return self.sdf(
            Q.rotation(
                query_positions[..., None, :].sub(self.translation), 
                self.orientation_conj
            )
        )


class SDFSmoothUnion(nn.Module):
    """
    Smooth union of multiple SDFs (signed distance functions).

    Args:
        sdfs (list of nn.Module): A list of callables that take a 3D tensor of query positions as input
                                  and return a tensor of signed distances.
        blend_k (float): A positive value that controls the smoothness of the union operation. 
                         Higher values result in a smoother union.

    Returns:
        A tensor of signed distances representing the smooth union of the input SDFs.
    """
    def __init__(
        self,
        sdfs: nn.Module,
        blend_k: float
    ):
        super().__init__()
        self.sdfs = nn.ModuleList(sdfs)
        self.register_buffer('blend_k', torch.tensor(blend_k))

    def forward(
        self,
        query_coords: Tensor
    ) -> Tensor:
        return (
            torch.cat([sdf(query_coords) for sdf in self.sdfs], dim=-2)
            .mul(-self.blend_k).logsumexp(dim=-2).div(-self.blend_k)
        )


class SDFUnion(nn.Module):
    """
    Union of multiple SDFs (signed distance functions).

    Args:
        sdfs (list of nn.Module): A list of callables that take a 3D tensor of query positions as input
                                  and return a tensor of signed distances.

    Returns:
        A tensor of signed distances representing the union of the input SDFs.
    """
    def __init__(
        self,
        sdfs: list[nn.Module]
    ):
        super().__init__()
        self.sdfs = nn.ModuleList(sdfs)

    def forward(
        self,
        query_coords: Tensor
    ) -> Tensor:
        return (
            torch.cat([sdf(query_coords) for sdf in self.sdfs], dim=-2)
            .min(dim=-2).values
        )

    
class SDFRounding(nn.Module):
    """
    Rounding operation of an SDF (signed distance function) defined over 3D space.

    Args:
        sdf (nn.Module): A callable that takes a 3D tensor of query positions as input and
                         returns a tensor of signed distances.
        rounding (float): The distance at which to perform the rounding.

    Returns:
        A tensor of signed distances after the rounding operation has been applied.
    """
    def __init__(
        self,
        sdf: nn.Module,
        rounding: float
    ):
        super().__init__()
        self.sdf = sdf
        self.register_buffer('rounding', torch.tensor(rounding))

    def forward(
        self,
        query_coords: Tensor
    ) -> Tensor:
        return self.sdf(query_coords).subtract(self.rounding)


class SDFOnion(nn.Module):
    def __init__(
        self,
        sdf: nn.Module,
        radii: float
    ):
        super().__init__()
        self.sdf = sdf
        self.register_buffer('radii',  torch.tensor(radii))

    def forward(
        self,
        query_coords: Tensor
    ) -> Tensor:
        return self.sdf(query_coords).abs().sub(self.radii)
