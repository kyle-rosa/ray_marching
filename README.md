# Ray Marching
This repository is a PyTorch based implementation of the sphere tracing algorithm for rendering geometry described by signed distance functions (SDFs).

## Quaternions and 3D Rotations
All SDFs and sensors are defined in their own reference frames.
The embedding of each object in 3D space is modified by applying affine transformations.
Affine transformations are represented by a quaternion that acts by rotation and a translation vector.

## Shaders
1. Lambertian.
2. Normals.
3. Polarisation:
4. Spin.

## User Input and Camera Control
Uses cv2 to poll keyboard inputs and pyautogui to poll mouse movements.
These are compiled into an affine transformation that is used to update the camera position and orientation for the next frame.

# Based on:
- Inigo Quilez: https://iquilezles.org/.