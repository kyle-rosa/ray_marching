import itertools

import torch

from control import RenderLoop, user_input_generator, user_input_mapper
from rendering.display import make_display_manager


torch.set_float32_matmul_precision(precision='high')


if __name__=='__main__':
    device = torch.device('cuda')
    dtype = torch.float32

    torch.set_default_dtype(dtype)

    render_loop = RenderLoop(
        num_cameras=1,
        px_width=800,
        px_height=800,
        focal_length=17e-3,
        sensor_width=17e-3,
        sensor_height=17e-3,
        marching_steps=32,
    ).to(device)
    render_loop = torch.compile(render_loop)#, mode='max-autotune')

    user_input = user_input_generator(device)
    input_mapper = user_input_mapper(device)
    display_manager = make_display_manager(
        window_width=800,
        window_height=800
    )

    user_input.send(None)
    input_mapper.send(None)
    display_manager.send(None)

    modes = ['lambertian', 'normal', 'tangent', 'spin']
    modes_cycle = itertools.cycle(modes)
    mode = next(modes_cycle)
    degree = torch.tensor(1).to(device=device)
    mode_index = {v: k for (k, v) in enumerate(modes)}

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
        (baseline, images) = render_loop(orientation_input, translation_input, degree)
        display_manager.send(images[mode_index[mode]].mul(baseline))
