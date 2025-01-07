import time

import torch
import torch.nn.functional as F

from control import EventAggregator, RenderLoop
from scene.scene_registry import make_test_scene, make_test_scene2

from torchwindow import Window
# import torchvision
# from pathlib import Path
import logging

torch._dynamo.config.verbose = True
torch._logging.set_logs(dynamo=logging.DEBUG)
torch.set_float32_matmul_precision(precision="medium")


if __name__ == "__main__":
    device = torch.device('cuda')
    dtype = torch.float16

    num_cameras = 1
    px_width, px_height = 1440, 900
    px_size = 3.45e-6
    marching_steps = 32

    scene = make_test_scene2().to(device, dtype)
    modes = [
        'lambertian', 'distance', 'proximity',
        'vignette', 'normal', 'laplacian',
        'tangent', 'spin'
    ]
    render_loop = RenderLoop(
        scene=scene,
        num_cameras=num_cameras,
        px_width=px_width,
        px_height=px_height,
        focal_length=(px_size * px_height),
        sensor_width=(px_size * px_width),
        sensor_height=(px_size * px_height),
        normals_eps=5e-2,
    ).to(device, dtype)
    render_loop = torch.compile(render_loop, mode='max-autotune')
    events = EventAggregator(
        initial_position=[(0., 0., 1.),],
        initial_orientation=[(1., 0., 0., 0.),],
        marching_steps=marching_steps
    ).to(device, dtype)
    window = Window(px_width, px_height, "Window")

    old_time = time.time()
    with torch.no_grad():
        while events.running:
            with torch.no_grad():
                (
                    positions,
                    orientations,
                    mode,
                    degree,
                    marching_steps,
                    save_frame
                ) = events.get_state()

            images = render_loop(
                orientations,
                positions,
                mode,
                degree,
                marching_steps,
            )
            # if save_frame:
            #     for key, image in images.items():
            #         torchvision.utils.save_image(
            #             image.movedim(-1, -3),
            #             Path(f'./output/{key}.png')
            #         )
            window.draw(
                F.pad(
                    images.mean(dim=0).float(),
                    pad=[0, 1],
                    value=1.0
                )
            )

            new_time = time.time()
            print(f'{1 / (new_time - old_time):.2f} frames per second')
            old_time = new_time
