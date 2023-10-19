from pathlib import Path

import pandas as pd
import pyautogui
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import quaternion as Q
from rendering.ray_marching import Marcher, PinholeCamera
from rendering.shader import SDFNormals, Shader

from pynput import mouse, keyboard
from collections import defaultdict


_default_device = torch.device('cpu')


class EventAggregator():
    def __init__(self):
        self.mouse_state = (0, 0)
        self.keyboard_state = defaultdict(bool)

        def on_move(x, y):
            self.mouse_state = (x, y)
        
        def on_click(x, y, button, pressed):
            pass

        def on_scroll(x, y, dx, dy):
            pass

        def on_press(key):
            self.keyboard_state[key] = True

        def on_release(key):
            self.keyboard_state[key] = False
        
        listeners = {
            'mouse': mouse.Listener(
                on_move=on_move,
                on_click=None,
                on_scroll=None
            ),
            'keyboard':  keyboard.Listener(
                on_press=on_press, 
                on_release=on_release
            )
        }
        self.controllers = {
            'mouse': mouse.Controller()
        }
        for listener in listeners.values():
            listener.start()

    def get_state(self):
        return self.mouse_state, self.keyboard_state


def user_input_generator(
    device: torch.device = _default_device,
    dtype: torch.dtype = torch.float32
):
    screen_size = (pyautogui.size().width//2, pyautogui.size().height)
    screen_centre = tuple(it//2 for it in screen_size)

    mu = torch.tensor(screen_centre, device=device, dtype=dtype)
    sigma = torch.tensor(screen_size, device=device, dtype=dtype).add(1).div(2)

    pyautogui.moveTo(*screen_centre)
    ndc_mouse_offset = torch.zeros(2, device=device, dtype=dtype)

    event_aggregator = EventAggregator()

    while True: 
        (mouse_state, keyboard_state) = event_aggregator.get_state()
        mouse_state = torch.tensor([mouse_state[0],  mouse_state[1]], device=device, dtype=dtype)
        ndc_mouse_offset = (mouse_state.sub(mu).div(sigma).clamp(-0.99, 0.99))
        keys = [ord(k.char) for (k, v) in keyboard_state.items() if (v and (type(k)==keyboard.KeyCode))]
        key = (keys[0] if keys else None)

        event_aggregator.controllers['mouse'].position = screen_centre

        yield (ndc_mouse_offset, key)


def get_keybindings(
    device: torch.device = _default_device,
    dtype: torch.dtype = torch.float32
):
    keybindings = pd.read_csv(Path() / 'data/keybindings.csv', header=0)
    keybindings['location_input'] = torch.from_numpy(keybindings[['X', 'Y', 'Z']].values).to(dtype).unbind(0)
    keybindings['orientation_input'] = torch.from_numpy(keybindings[['YZ', 'ZX', 'XY']].values).to(dtype).unbind(0)
    keybindings['location_input'] = keybindings['location_input'].apply(lambda x: x.to(device=device, dtype=dtype))
    keybindings['orientation_input'] = keybindings['orientation_input'].apply(lambda x: x.to(device=device, dtype=dtype))
    keybindings['ord'] = keybindings['key'].apply(ord)
    keybindings = keybindings.set_index('ord')
    return keybindings.T.to_dict()


def user_input_mapper(
    device: torch.device = _default_device,
    dtype: torch.dtype = torch.float32
):
    """
    Polls user input events from the mouse and keyboard and uses them to calculate
    position and orientation updates.
    """
    keybindings = get_keybindings(device=device, dtype=dtype)
    default_position_input = torch.zeros((1, 3), device=device, dtype=dtype)
    default_orientation_input = torch.zeros((1, 3), device=device, dtype=dtype)
    
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
        initial_orientation: list[tuple[float, float, float]] = [(1., 0., 0., 0.)],
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.register_buffer('position', torch.tensor(initial_position, dtype=dtype))
        self.register_buffer('orientation', torch.tensor(initial_orientation, dtype=dtype))

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
                Q.to_versor(orientation_input.div(4)).expand_as(self.orientation)
            ),
            p=2, dim=-1, eps=0
        )
        return (self.orientation, self.position)


class RenderLoop(nn.Module):
    def __init__(
        self,
        scene,
        num_cameras: int = 1,
        px_width: int = 800,
        px_height: int = 800,
        focal_length: float = 17e-3,
        sensor_width: float = 17e-3,
        sensor_height: float = 17e-3,
        marching_steps: int = 32,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.scene = scene
        self.integrator = ConfigurationIntegrator(
            initial_position=[(0., 0., 0.)],
            initial_orientation=[(1., 0., 0., 0.)],
            dtype=dtype
        )
        self.camera = PinholeCamera(
            num_cameras=num_cameras,
            px_width=px_width,
            px_height=px_height,
            focal_length=focal_length,
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            dtype=dtype
        )
        self.marcher = Marcher(
            sdf_scene=self.scene,
            marching_steps=marching_steps,
        )
        self.shader = Shader(
            cyclic_cmap=torch.load(Path() / 'data/cyclic_cmap.pt'),
            decay_factor=0.01,
            dtype=dtype
        )
        self.normals = SDFNormals(
            sdf_scene=self.scene,
            dtype=dtype
        )


    def forward(
        self,
        orientation_input: Tensor,
        translation_input: Tensor,
        degree: int = 1
    ):
        (orientations, translations) = self.integrator(orientation_input, translation_input)
        (pixel_pos, pixel_frames, ray_pos, ray_dirs) = self.camera(orientations, translations)
        marched_ray_pos = self.marcher(ray_pos, ray_dirs)
        surface_normals = self.normals(marched_ray_pos)
        return self.shader(
            pixel_pos, orientations, pixel_frames, 
            ray_dirs, marched_ray_pos, surface_normals,
            degree=degree
        )
