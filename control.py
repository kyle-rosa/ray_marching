from pathlib import Path

import cv2
import pandas as pd
import pyautogui
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import quaternion as Q
from rendering.ray_marching import Marcher, PinholeCamera
from rendering.shader import SDFNormals, Shader
from scene.scene_registry import make_test_scene


_default_device = torch.device('cpu')

def user_input_generator(
    device: torch.device = _default_device
):
    screen_size = (pyautogui.size().width//2, pyautogui.size().height)
    screen_centre = tuple(it//2 for it in screen_size)

    mu = torch.tensor(screen_centre, device=device)
    sigma = torch.tensor(screen_size, device=device).add(1).div(2)

    pyautogui.moveTo(*screen_centre)
    ndc_mouse_offset = torch.zeros(2, device=device)
    
    while True: 
        key = cv2.waitKey(1) & 0xFF
        yield (ndc_mouse_offset, key)

        pos = pyautogui.position()
        ndc_mouse_offset = torch.tensor([pos.x, pos.y], device=device, dtype=torch.float32).sub(mu).div(sigma).clamp(-0.99, 0.99)
        pyautogui.moveTo(*screen_centre, _pause=False)


def get_keybindings(
    device: torch.device = _default_device,
):
    keybindings = pd.read_csv(Path() / 'data/keybindings.csv', header=0)
    keybindings['location_input'] = torch.from_numpy(keybindings[['X', 'Y', 'Z']].values).float().unbind(0)
    keybindings['orientation_input'] = torch.from_numpy(keybindings[['YZ', 'ZX', 'XY']].values).float().unbind(0)
    keybindings['location_input'] = keybindings['location_input'].apply(lambda x: x.to(device=device))
    keybindings['orientation_input'] = keybindings['orientation_input'].apply(lambda x: x.to(device=device))
    keybindings['ord'] = keybindings['key'].apply(ord)
    keybindings = keybindings.set_index('ord')
    return keybindings.T.to_dict()


def user_input_mapper(
    device: torch.device = _default_device
):
    keybindings = get_keybindings(device=device)
    default_position_input = torch.zeros((1, 3), device=device)
    default_orientation_input = torch.zeros((1, 3), device=device)
    
    position_input = default_position_input
    orientation_input = default_orientation_input
    while True:
        (ndc_mouse_offset, key) = yield (orientation_input, position_input)

        mouse_orientation_input = F.pad(ndc_mouse_offset, [0, 1], value=0.)
        mouse_orientation_input = mouse_orientation_input[..., [1, 0, 2]]
        if key in keybindings:
            keyboard_position_input = keybindings[key]['location_input'][None]
            keyboard_orientation_input =  keybindings[key]['orientation_input'][None]
        else: 
            keyboard_position_input = default_position_input
            keyboard_orientation_input = default_orientation_input

        position_input = keyboard_position_input
        orientation_input = mouse_orientation_input.add(keyboard_orientation_input)


class ConfigurationIntegrator(nn.Module):
    def __init__(
        self,
        initial_position: list[tuple[float, float, float]] = [(0., 0., 0.)],
        initial_orientation: list[tuple[float, float, float]] = [(1., 0., 0., 0.)]
    ):
        super().__init__()
        self.register_buffer('position', torch.tensor(initial_position))
        self.register_buffer('orientation', torch.tensor(initial_orientation))

    def forward(
        self,
        orientation_input: Tensor,
        translation_input: Tensor
    ) -> tuple[Tensor, Tensor]:
        self.position = Q.rotation(
            translation_input.div(10).expand_as(self.position),
            self.orientation
        ).add(self.position)
        self.orientation = F.normalize(
            Q.multiply(
                self.orientation,
                Q.to_versor(orientation_input.div(10)).expand_as(self.orientation)
            ),
            p=2, dim=-1, eps=0
        )
        return (self.orientation, self.position)


class RenderLoop(nn.Module):
    def __init__(
        self,
        num_cameras: int = 1,
        px_width: int = 800,
        px_height: int = 800,
        focal_length: float = 17e-3,
        sensor_width: float = 17e-3,
        sensor_height: float = 17e-3,
        marching_steps: int = 32,
    ):
        super().__init__()
        self.scene = make_test_scene()
        self.integrator = ConfigurationIntegrator(
            initial_position=[(0., 0., 0.)],
            initial_orientation=[(1., 0., 0., 0.)],
        )
        self.camera = PinholeCamera(
            num_cameras=num_cameras,
            px_width=px_width,
            px_height=px_height,
            focal_length=focal_length,
            sensor_width=sensor_width,
            sensor_height=sensor_height,
        )
        self.marcher = Marcher(
            sdf_scene=self.scene,
            marching_steps=marching_steps
        )
        self.shader = Shader(
            cyclic_cmap=torch.load(Path() / 'data/cyclic_cmap.pt'),
            decay_factor=0.01
        )
        self.normals = SDFNormals(
            sdf_scene=self.scene
        )


    def forward(
        self,
        orientation_input: Tensor,
        translation_input: Tensor,
        degree: int = 1
    ):
        orientations, translations = self.integrator(orientation_input, translation_input)
        (pixel_pos, pixel_frames, ray_pos, ray_dirs) = self.camera(orientations, translations)
        marched_ray_pos = self.marcher(ray_pos, ray_dirs)
        surface_normals = self.normals(marched_ray_pos)
        return self.shader(
            pixel_pos, orientations, pixel_frames, 
            ray_dirs, marched_ray_pos, surface_normals,
            degree=degree
        )
