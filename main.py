import itertools
import pprint
import time

import torch

from control import RenderLoop, user_input_generator, user_input_mapper
from rendering.display import make_display_manager
from scene.scene_registry import make_simple_scene, make_test_scene


torch.set_float32_matmul_precision(precision='high')


if __name__=='__main__':
    device = torch.device('cuda')
    dtype = torch.float16

    px_width = 1_440
    px_height = 900

    px_size = 3.45e-6

    scene = make_test_scene(dtype=dtype)
    render_loop = RenderLoop(
        scene=scene,
        num_cameras=1,
        px_width=px_width,
        px_height=px_height,
        focal_length=(px_size * px_height),
        sensor_width=(px_size * px_width),
        sensor_height=(px_size * px_height),
        marching_steps=32,
        dtype=dtype,
    ).to(device)
    render_loop = torch.compile(render_loop)
    # render_loop = torch.compile(render_loop, mode='max-autotune')

    user_input = user_input_generator(device=device, dtype=dtype)
    input_mapper = user_input_mapper(device=device, dtype=dtype)
    display_manager = make_display_manager(window_width=px_width, window_height=px_height, device=device)

    user_input.send(None)
    input_mapper.send(None)
    display_manager.send(None)

    modes = ['lambertian', 'normal', 'tangent', 'spin']
    modes_cycle = itertools.cycle(modes)
    mode = next(modes_cycle)
    degree = torch.tensor(1).to(device=device)
    mode_index = {v: k for (k, v) in enumerate(modes)}

    # optimizer = torch.optim.AdamW(params=render_loop.parameters(), lr=0.001)

    old_time = time.time()
    while True:
        # optimizer.zero_grad()
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

        # loss = images[-1].var().sum()
        # loss.backward()
        # optimizer.step()

        with torch.no_grad():
            render = images[mode_index[mode]].mul(baseline)
            display_manager.send(render)

        new_time = time.time()
        print( f'{round(1 / (new_time - old_time), 2)} frames per second' )
        old_time = new_time
