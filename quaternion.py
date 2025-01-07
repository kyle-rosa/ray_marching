import torch
import torch.nn as nn
from torch import Tensor


def cross_product(U: Tensor, V: Tensor) -> Tensor:
    """
    Computes the cross product between two 3D tensors U and V along their last axis.

    Args:
        U (Tensor): A 3D tensor of shape (..., 3).
        V (Tensor): A 3D tensor of shape (..., 3).

    Returns:
        Tensor: A 3D tensor of the same shape as U and V, representing the cross product
        of U and V.
    """
    return (
        U[..., [1, 2, 0]].mul(V[..., [2, 0, 1]])
        .sub(U[..., [2, 0, 1]].mul(V[..., [1, 2, 0]]))
    )


def multiply(p: Tensor, q: Tensor) -> Tensor:
    """
    Computes the product of two 4D tensors p and q, representing quaternions.

    Args:
        p (torch.Tensor): A 4D tensor of shape (..., 4), representing the first
        quaternion.
        q (torch.Tensor): A 4D tensor of shape (..., 4), representing the second
        quaternion.

    Returns:
        torch.Tensor: A 4D tensor of the same shape as p and q, representing the product
        of p and q.
    """
    return torch.stack(
        [
            p[..., 0] * q[..., 0] - (p[..., 1:] * q[..., 1:]).sum(-1),
            (p[..., [0, 1, 2]] * q[..., [1, 0, 3]]).sum(-1) - p[..., 3] * q[..., 2],
            (p[..., [0, 2, 3]] * q[..., [2, 0, 1]]).sum(-1) - p[..., 1] * q[..., 3],
            (p[..., [0, 1, 3]] * q[..., [3, 2, 0]]).sum(-1) - p[..., 2] * q[..., 1],
        ],
        dim=-1
    )


def conjugate(q: Tensor) -> Tensor:
    return q.mul(
        torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)
    )


def rotation(V: Tensor, q: Tensor) -> Tensor:
    """
    Computes the rotation of a 3D vector V by a quaternion q.

    Args:
        V (torch.Tensor): A 3D tensor of shape (..., 3), representing the vector to be
            rotated.
        q (torch.Tensor): A 4D tensor of shape (..., 4), representing the quaternion
            defining the rotation.

    Returns:
        torch.Tensor: A 3D tensor of the same shape as V, representing the rotated
        vector.
    """
    qv = q[..., 1:]
    t = cross_product(qv, V).mul(2)
    y = cross_product(qv, t)
    return y.add(q[..., [0]].mul(t)).add(V)


def to_versor(V: Tensor) -> Tensor:
    """
    Converts a 3D vector V into a versor, i.e., a unit vector with an additional scalar
    component of 1.

    Args:
        V (torch.Tensor): A 3D tensor of shape (..., 3), representing the vector to be
        converted.

    Returns:
        torch.Tensor: A 4D tensor of shape (..., 4), representing the versor
        corresponding to V.
    """
    return torch.cat(
        [V.pow(2).sum(dim=-1, keepdim=True).neg().add(1).pow(0.5), V],
        dim=-1
    )


class QuaternionToSO3(nn.Module):
    """
    A PyTorch module that converts quaternions to rotation matrices in SO(3).

    Attributes:
        pairs (torch.Tensor): A 2D tensor of shape (10, 2), representing the indices of
        pairs of elements to be multiplied to compute the rotation matrix from the
        quaternion.

    Methods:
        forward(q: torch.Tensor) -> torch.Tensor:
            Computes the rotation matrix corresponding to the input quaternion q.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer(
            'pairs',
            torch.tensor([(i, j) for i in range(4) for j in range(i, 4)])
        )

    def forward(self, q: Tensor) -> Tensor:
        (ww, wx, wy, wz, xx, xy, xz, yy, yz, zz) = (
            q[..., self.pairs[:, 0]].mul(q[..., self.pairs[:, 1]]).unbind(-1)
        )
        return torch.stack(
            [
                (ww + xx - yy - zz), 2 * (xy - wz), 2 * (wy + xz),
                2 * (xy + wz), (ww - xx + yy - zz), 2 * (yz - wx),
                2 * (xz - wy), 2 * (wx + yz), (ww - xx - yy + zz),
            ], dim=-1
        ).view(-1, 3, 3)
