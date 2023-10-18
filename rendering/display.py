import torch

from torchwindow import Window

def make_display_manager(
    window_width: int,
    window_height: int,
    device: torch.device
) -> None:
    """
    Coroutine that creates a window for displaying and saving videos. 
    
    Args:
        window_width (int): The width of the display window.
        window_height (int): The height of the display window.
        device (torch.device): The CUDA device on which images will be read from.
    
    Yields:
        None
    """
    window = Window(window_width, window_height, "Renderer")
    buffer = torch.ones((window_height, window_width, 4), dtype=torch.float, device=device)
    while True:
        display_img = yield
        buffer[..., :3] = display_img[0].float()
        window.draw(buffer)
