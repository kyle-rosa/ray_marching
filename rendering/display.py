import torch
import cv2


def to_numpy_byte_image(tensor_image):
    return (
        tensor_image
        .multiply(256).floor().clamp(0, 255).to(torch.uint8)
        .flip(dims=[-2]).to('cpu').numpy()
    )


def make_display_manager(window_width, window_height):
    window = 'render'
    writers = {}
    cv2.startWindowThread()
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    writers[window] = cv2.VideoWriter(
        f'{window}.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 30, (window_height, window_width)
    )
    while True:
        images_tensor = yield
        images = to_numpy_byte_image(images_tensor.expand(-1, -1, -1, 3))[0]
        cv2.imshow(window, images[..., [2, 1, 0]])
        writers[window].write(images[..., [2, 1, 0]])
