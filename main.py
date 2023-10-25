import itertools
import pprint
import time

import torch
import torch.nn.functional as F

from control import EventAggregator, RenderLoop
from scene.scene_registry import make_test_scene, make_test_scene2

from torchwindow import Window
import torchvision
from pathlib import Path


torch.set_float32_matmul_precision(precision='highest')


if __name__=='__main__':
    device = torch.device('cuda')
    dtype = torch.float16

    px_width = 1_280
    px_height = 720
    px_size = 3.45e-6     
    marching_steps = 32
    legs = 1

    # scene = make_test_scene(dtype=dtype)
    scene = make_test_scene2().to(device, dtype)
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
    ).to(device, dtype)
    render_loop = torch.compile(render_loop)
    # render_loop = torch.compile(render_loop, mode='max-autotune')

    events = EventAggregator().to(device, dtype)
    window = Window(px_width, px_height, "Renderer")
    modes = [
        'lambertian', 'distance', 'proximity',
        'vignette', 'normal', 'laplacian',
        'tangent', 'spin'
    ]

    # optimizer = torch.optim.AdamW(params=render_loop.parameters(), lr=0.001)

    old_time = time.time()
    with torch.no_grad():
        while events.running:
            # optimizer.zero_grad()
            (positions, orientations, mode, degree, marching_steps, save_frame) = events.get_state()
            images = render_loop(orientations, positions, degree, marching_steps, legs)
            render = images[modes[mode % len(modes)]]

            if save_frame:
                for key, image in images.items():
                    torchvision.utils.save_image(image.movedim(-1, -3), Path() / f'output/{key}.png')

            # loss = images[-1].var().sum()
            # loss.backward()
            # optimizer.step()
            
            # render_large = F.interpolate(render.movedim(-1, -3), scale_factor=(2, 2)).movedim(-3, -1)
            window.draw(F.pad(render[0].float(), [0, 1], value=1.0))

            new_time = time.time()
            print( f'{round(1 / (new_time - old_time), 2)} frames per second' )
            old_time = new_time
