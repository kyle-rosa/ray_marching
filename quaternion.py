import torch
import torch.nn as nn


def cross_product(U, V):
    return (
        U[..., [1, 2, 0]].mul(V[..., [2, 0, 1]])
        .sub(U[..., [2, 0, 1]].mul(V[..., [1, 2, 0]]))
    )


def multiplication(p, q):
    return torch.stack(
        [
            p[..., 0] * q[..., 0] - (p[..., 1:] * q[..., 1:]).sum(-1),
            (p[..., [0, 1, 2]] * q[..., [1, 0, 3]]).sum(-1) - p[..., 3] * q[..., 2],
            (p[..., [0, 2, 3]] * q[..., [2, 0, 1]]).sum(-1) - p[..., 1] * q[..., 3],
            (p[..., [0, 1, 3]] * q[..., [3, 2, 0]]).sum(-1) - p[..., 2] * q[..., 1],
        ],
        dim=-1
    )


def conjugate(q):
    return torch.cat([q[..., [0]], q[..., 1:].neg()], dim=-1)


def rotation(V, q):
    qv = q[..., 1:]#.expand_as(V)
    t = cross_product(qv, V).mul(2)
    y = cross_product(qv, t)
    return y.add(q[..., [0]].mul(t)).add(V)


def to_versor(V):
    return torch.cat([V.pow(2).sum(dim=-1, keepdim=True).neg().add(1).pow(0.5), V], dim=-1)


class QuaternionToSO3(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('pairs', torch.tensor([(i, j) for i in range(4) for j in range(i, 4)]))

    def forward(self, q):
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
