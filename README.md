# Ray Marching
<p align="center">
  <img src="gallery/lambertian.png?raw=true" width="200"> 
  <img src="gallery/normal.png?raw=true" width="200"> 
  <img src="gallery/tangent.png?raw=true" width="200"> 
  <img src="gallery/spin.png?raw=true" width="200">
</p>

*Ray marching* is a method for rendering computer graphics. 
Ray marching algorithms operate on a *signed distance field* (SDF) representation of the scene, instead of textured meshes. SDFs encode the distance between points in space and the surfaces of the various objects in the scene. This allows one to 




This repository is a PyTorch based implementation of the sphere tracing algorithm for rendering geometry described by signed distance functions (SDFs).

## Quaternions and 3D Rotations
The embedding of each object in 3D space is modified by applying affine transformations, which are represented by a translation vector and a rotation encoded as a quaternion.

## Shaders
1. Lambertian: Very simple geometric illumination model.
<!-- ![alt text](https://github.com/kyle-rosa/ray_marching/blob/main/gallery/lambertian.png?raw=true) -->
2. Normals: Surfaces are coloured based on the coordinates of their normal vectors.
<!-- ![alt text](https://github.com/kyle-rosa/ray_marching/blob/main/gallery/normal.png?raw=true) -->
3. Tangent: Surface normals are projected onto the camera sensor, treated as complex numbers and domain-coloured.
<!-- ![alt text](https://github.com/kyle-rosa/ray_marching/blob/main/gallery/tangent.png?raw=true) -->
4. Spin: Surfaces are coloured based on a combination of their normal vector and the quaternions that define the embeddings of the camera and object in world space. This shader mimics the behaviour of spin-1/2 objects in physics --- rotating the camera 360 degrees reverses the orientation of the texture on the surface.
<!-- ![alt text](https://github.com/kyle-rosa/ray_marching/blob/main/gallery/spin.png?raw=true) -->


## User Input and Camera Control
Uses cv2 to poll keyboard inputs and pyautogui to poll mouse movements.
These are compiled into an affine transformation that is used to update the camera position and orientation for the next frame.

# Based on:
- Inigo Quilez: https://iquilezles.org/.