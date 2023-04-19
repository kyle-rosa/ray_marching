import itertools

import torch
import torch.nn.functional as F

from control import GameLoop, user_input_generator, user_input_mapper
from rendering.display import make_display_manager

torch.set_float32_matmul_precision(precision='high')


if __name__=='__main__':
    device = torch.device('cuda')
    dtype = torch.float32

    game_loop = GameLoop().to(device)
    game_loop = torch.compile(game_loop)

    user_input = user_input_generator(dtype, device)
    input_mapper = user_input_mapper(dtype, device)
    display_manager = make_display_manager(800, 800)

    user_input.send(None)
    input_mapper.send(None)
    display_manager.send(None)


    modes = ['lambertian', 'normal', 'tangent', 'spin']
    modes_cycle = itertools.cycle(modes)
    mode = next(modes_cycle)
    degree = torch.tensor(1).to(dtype=dtype, device=device)
    mode_index = {v: k for (k, v) in enumerate(modes)}

    orientations = F.normalize(torch.randn((1, 1, 1, 4)), p=2, dim=-1, eps=0).to(device)
    translations = torch.randn((1, 1, 1, 3)).to(device)

    while True:
        (ndc_mouse_offset, key) = user_input.send(None)

        if key == ord("q"):
            break
        if key == ord("m"):
            mode = next(modes_cycle)
        if key == ord('i'):
            degree = degree.add(1)
        if key == ord('o'):
            degree = degree.add(-1)

        (orientation_input, translation_input) = input_mapper.send((ndc_mouse_offset, key))
        (baseline, images) = game_loop(orientation_input, translation_input, degree)
        display_manager.send(images[mode_index[mode]].mul(baseline))
