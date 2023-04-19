import cv2
import torch
from torch import Tensor
from pathlib import Path
import numpy as np

def to_numpy_byte_image(
    tensor_image: Tensor
) -> np.ndarray:
    """
    Converts a torch tensor image to a numpy uint8 array.

    Args:
        tensor_image (torch.Tensor): A torch tensor of shape (batch_size, num_channels, height, width).

    Returns:
        np.ndarray: A numpy uint8 array of shape (batch_size, height, width, num_channels).
    """
    return (
        tensor_image
        .multiply(256).floor().clamp(0, 255)
        .to(torch.uint8).to('cpu').numpy()
    )


def make_display_manager(
    window_width: int,
    window_height: int
) -> None:
    """
    Coroutine that creates a window for displaying and saving videos. 
    It receives a stream of images as input, which it then displays on the window and writes to disk. 
    
    Args:
        window_width (int): The width of the display window.
        window_height (int): The height of the display window.
    
    Yields:
        None
    """
    window = 'render'
    writers = {}
    cv2.startWindowThread()
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    writers[window] = cv2.VideoWriter(
        str(Path() / f'output/{window}.mp4'), 
        cv2.VideoWriter_fourcc(*"mp4v"), 
        20.0, 
        (window_height, window_width)
    )
    while True:
        images_tensor = yield
        images = to_numpy_byte_image(images_tensor.flip(dims=[-2]).expand(-1, -1, -1, 3))[0]
        cv2.imshow(window, images[..., [2, 1, 0]])
        writers[window].write(images[..., [2, 1, 0]])
