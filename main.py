import itertools
import pprint
import time

import torch
import torch.nn.functional as F

from control import RenderLoop, user_input_generator, user_input_mapper
from scene.scene_registry import make_test_scene

from torchwindow import Window


torch.set_float32_matmul_precision(precision='high')


if __name__=='__main__':
    device = torch.device('cuda')
    dtype = torch.float32

    px_width = 1_280
    px_height = 720
    px_size = 3.45e-6
    marching_steps = 32

    scene = make_test_scene(dtype=dtype)
    render_loop = RenderLoop(
        scene=scene,
        num_cameras=1,
        px_width=px_width,
        px_height=px_height,
        focal_length=(px_size * px_height),
        sensor_width=(px_size * px_width),
        sensor_height=(px_size * px_height),
        marching_steps=marching_steps,
        normals_eps=5e-2,
        dtype=dtype,
    ).to(device)
    render_loop = torch.compile(render_loop)
    # render_loop = torch.compile(render_loop, mode='max-autotune')

    user_input = user_input_generator(device=device, dtype=dtype)
    input_mapper = user_input_mapper(device=device, dtype=dtype)
    
    window = Window(px_width * 2, px_height * 2, "Renderer")

    user_input.send(None)
    input_mapper.send(None)

    modes = [
        'lambertian', 'distance', 'proximity',
        'vignette', 'normal', 'laplacian',
        'tangent', 'spin'
    ]
    modes_cycle = itertools.cycle(modes)
    mode = next(modes_cycle)
    degree = torch.tensor(1).to(device=device)
    mode_index = {v: k for (k, v) in enumerate(modes)}

    # optimizer = torch.optim.AdamW(params=render_loop.parameters(), lr=0.001)

    old_time = time.time()
    with torch.no_grad():
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
            (
                lambertian_layer,
                distance_layer,
                proximity_layer,
                vignette_layer,
                normal_layer,
                laplacian_layer,
                tangent_layer,
                spin_layer
            ) = render_loop(orientation_input, translation_input, degree)
            render = (
                vignette_layer
                .add(laplacian_layer)
                .div(2)
                .mul(vignette_layer)
            )[..., [0, 0, 0]]

            # loss = images[-1].var().sum()
            # loss.backward()
            # optimizer.step()
            
            render_large = F.interpolate(render.movedim(-1, -3), scale_factor=(2, 2)).movedim(-3, -1)
            window.draw(F.pad(render_large[0].float(), [0, 1], value=1.0))

            new_time = time.time()
            print( f'{round(1 / (new_time - old_time), 2)} frames per second' )
            old_time = new_time
