from pathlib import Path

import pandas as pd
import pyautogui
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import quaternion as Q
from rendering.ray_marching import (PinholeCamera, SDFMarcher, SDFNormals)
from rendering.shader import Shader

from pynput import (mouse, keyboard)
from collections import defaultdict


class EventAggregator(nn.Module):
    def __init__(
        self,
        initial_position: list[tuple[float, float, float]],
        initial_orientation: list[tuple[float, float, float]],
        marching_steps: int = 32,
    ):
        super().__init__()

        self.register_buffer('position', torch.tensor(initial_position))
        self.register_buffer('orientation', torch.tensor(initial_orientation))
        self.translation_sensitivity = 0.1
        self.rotation_sensitivity = 0.25

        self.screen_size = (pyautogui.size().width//2, pyautogui.size().height)
        self.screen_centre = tuple(it//2 for it in self.screen_size)

        # Initialise state variables:
        self.mouse_state = self.screen_centre
        self.keyboard_state = defaultdict(bool)
        self.marching_steps = marching_steps
        self.mode = 0
        self.degree = 2
        self.save_frame = False

        # Define input events:
        def on_move(x, y):
            self.mouse_state = (x, y)

        def on_click(x, y, button, pressed):
            pass

        def on_scroll(x, y, dx, dy):
            if (dy > 0):
                self.mode = self.mode + 1
            if (dy < 0):
                self.mode = self.mode - 1

        def on_press(key):
            self.keyboard_state[key] = True
            if isinstance(key, keyboard.KeyCode):
                if key.char == 'q':
                    self.running = False
                if key.char == 'i':
                    self.degree = self.degree + 1
                if key.char == 'o':
                    self.degree = self.degree - 1
                if key.char == 'm':
                    self.marching_steps = self.marching_steps + 1
                if key.char == 'n':
                    self.marching_steps = self.marching_steps - 1
                if key.char == 'p':
                    self.save_frame = True

        def on_release(key):
            self.keyboard_state[key] = False

        self.running = True
        self.listeners = {
            'mouse': mouse.Listener(
                on_move=on_move,
                on_click=on_click,
                on_scroll=on_scroll
            ),
            'keyboard':  keyboard.Listener(
                on_press=on_press,
                on_release=on_release
            )
        }
        for listener in self.listeners.values():
            listener.start()
        self.controllers = {
            'mouse': mouse.Controller()
        }
        self.controllers['mouse'].position = self.screen_centre
        self.register_buffer('ndc_mouse_diff', torch.tensor((0., 0.)))

        self.kb_dict = (
            pd.read_csv(Path('./data/key_bindings.csv'), header=0)
            .set_index('key').T.to_dict()
        )
        self.translations_mappings = nn.ParameterDict({
            key: nn.Parameter(
                torch.tensor(
                    [self.kb_dict[key][it] for it in "XYZ"])
            ) for key in self.kb_dict
        })
        self.orientations_mappings = nn.ParameterDict({
            key: nn.Parameter(
                torch.tensor([self.kb_dict[key][it] for it in ["YZ", "ZX", "XY"]])
            ) for key in self.kb_dict
        })

        self.register_buffer('default_position_input', torch.tensor([[0., 0., 0.]]))
        self.register_buffer('default_orientation_input', torch.tensor([[0., 0., 0.]]))

    def get_state(self):
        # Get mouse offsets:
        self.ndc_mouse_diff[0] = (
            (self.mouse_state[0] - self.screen_centre[0])
            / self.screen_centre[0]
        )
        self.ndc_mouse_diff[1] = (
            (self.mouse_state[1] - self.screen_centre[1])
            / self.screen_centre[1]
        )
        # self.controllers['mouse'].position = self.screen_centre

        # Get key presses:
        keys = [
            k.char
            for (k, v) in self.keyboard_state.items()
            if (v and isinstance(k, keyboard.KeyCode))
        ]
        key = (keys[0] if keys else None)

        # Convert mouse input to components:
        mouse_orientation_input = F.pad(self.ndc_mouse_diff, [0, 1], value=0.)
        mouse_orientation_input = mouse_orientation_input[..., [1, 0, 2]]

        # Convert keyboard input to components:
        keyboard_position_input = self.default_position_input.clone()
        keyboard_orientation_input = self.default_orientation_input.clone()
        for key in set(self.kb_dict).intersection(keys):
            keyboard_position_input += self.translations_mappings[key][None]
            keyboard_orientation_input += self.orientations_mappings[key][None]

        # Add mouse and keyboard effects together:
        translation_input = keyboard_position_input
        orientation_input = mouse_orientation_input.add(keyboard_orientation_input)

        # Apply updates to position and orientation parameters:
        self.position = Q.rotation(
            (
                translation_input
                .mul(self.translation_sensitivity)
                .expand_as(self.position)
            ),
            self.orientation
        ).add(self.position)
        self.orientation = F.normalize(
            Q.multiply(
                self.orientation,
                Q.to_versor(
                    orientation_input.mul(self.rotation_sensitivity)
                ).expand_as(self.orientation)
            ), p=2, dim=-1, eps=0
        )

        save_frame = self.save_frame
        self.save_frame = False
        return (
            self.position,
            self.orientation,
            self.mode,
            self.degree,
            self.marching_steps,
            save_frame
        )


def make_reflection_directions(
    surface_normals: Tensor,
    ray_dirs: Tensor,
):
    normal_projections = (
        surface_normals
        .mul(ray_dirs.mul(-1))
        .sum(dim=-1, keepdim=True)
        .mul(surface_normals)
    )
    reflected_dirs = (
        normal_projections
        .mul(2)
        .add(ray_dirs)
    )
    return reflected_dirs


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
        normals_eps: float = 5e-2,
    ):
        super().__init__()
        self.scene = scene

        self.px_width = px_width
        self.px_height = px_height
        self.camera = PinholeCamera(
            num_cameras=num_cameras,
            px_width=px_width,
            px_height=px_height,
            focal_length=focal_length,
            sensor_width=sensor_width,
            sensor_height=sensor_height,
        )
        self.marcher = SDFMarcher(
            sdf_scene=self.scene,
        )
        self.normals = SDFNormals(
            sdf_scene=self.scene,
            normals_eps=normals_eps,
        )
        self.shader = Shader()

    def forward(
        self,
        orientations: Tensor,
        translations: Tensor,
        mode: int = 0,
        degree: int = 1,
        marching_steps: int = 32,
    ):
        (pixel_pos, pixel_frames, ray_pos, ray_dirs) = self.camera(
            orientation=orientations,
            translation=translations
        )
        marched_ray_pos = self.marcher(ray_pos, ray_dirs, marching_steps)
        surface_distances = self.scene(marched_ray_pos)
        (surface_normals, surface_laplacian) = self.normals(marched_ray_pos)
        image = self.shader(
            pixel_pos,
            orientations,
            pixel_frames,
            ray_dirs,
            marched_ray_pos,
            surface_normals,
            surface_laplacian,
            surface_distances,
            mode=mode,
            degree=degree,
        ).expand(-1, self.px_height, self.px_width, 3)
        return image
