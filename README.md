# Ray Marching
<p align="center">
  <img src="gallery/lambertian.png?raw=true" width="200"> 
  <img src="gallery/normal.png?raw=true" width="200"> 
  <img src="gallery/tangent.png?raw=true" width="200"> 
  <img src="gallery/spin.png?raw=true" width="200">
</p>

# Introduction
*Ray marching* is a method for rendering computer graphics.
In contrast with other rendering methods that use textured meshes, ray marching algorithms operate on a *signed distance field* (SDF) representation of the scene. This repository contains the following: 
1. A constructive geometry system for building complex SDFs by composing simple primitives.
2. Ray marching operations for rendering SDFs.
3. A collection of geometrically motivated shaders.
4. An interactive first person control system for the camera that uses mouse and keyboard inputs.

### Differentiability
All scene construction and rendering functions are implemented in PyTorch and are fully differentiable. This allows us to optimise the scene parameters based on the pixel values of the rendered output image.

# Design and Implementation Details
## SDF Constructive Geometry System
I've implemented a rudimentary constructive geometry system for signed distance functions, largely based on the work in reference [1]. This includes: 
1. A collection of primitive SDFs for simple geometric shapes: 
    1. spheres,
    2. rectangular prisms,
    3. planes, 
    4. lines, 
    5. disks, and
    6. tori.
2. Ways to transform individual SDFs: 
    1. affine transformations (rotations and reflections), 
    2. rounding, and
    3. layering.
3. Methods to combine SDFs into new ones:
    1. union, and
    2. smooth union.
 
### Quaternions and 3D Rotations
Wherever possible, I've encoded rotations with quaternions. 

The space of 3D rotations has a ``hole" in it. Paths through the space of 3D rotations can become knotted on this hole and negatively impact the result of gradient descent algorithms, while using quaternions avoids this. 

Using quaternions everywhere also allows us to track the spin orientation between objects as they move around the scene. Going further, we can design shaders that actually render objects differently based on their spin orientation relative to the camera.

## Ray Tracing
### Walk on Spheres
Given a signed distance function $f$, we can trace a ray from position $x_0$ in direction $v_0$ using the formula $x_{i+1} = x_i + f(x_i) \times v_0$. If the ray intersects the surface described by $f$, then $x_i$ will approach the intersection point on the surface.

### Surface Normal Calculation

Once we've found a ray-surface intersection point $p$, we can calculate the surface normal vector at $p$ by querying the SDF at nearby points and estimating the gradient numerically. Alternatively, it's possible to find the surface normal vector by back-propagating the value of $f(p)$ and inspecting the gradient of $p$.

## Shaders
### Lambertian Shader
Very simple geometric illumination model.
<p align="center">
  <img src="gallery/lambertian.png?raw=true" width="400"> 
</p>

### Normals Shader
Surfaces are coloured based on the coordinates of their normal vectors.
<p align="center">
  <img src="gallery/normal.png?raw=true" width="400"> 
</p>

### Tangents Shader
Surface normals are projected onto the camera sensor, treated as complex numbers and domain-coloured.
<p align="center">
  <img src="gallery/tangent.png?raw=true" width="400"> 
</p>

### Spin Shader
Surfaces are coloured based on a combination of their normal vector and the quaternions that define the embeddings of the camera and object in world space. This shader mimics the behaviour of spin-$1/2$ objects in physics --- rotating the camera $360$ degrees reverses the orientation of the texture on the surface.
<p align="center">
  <img src="gallery/spin.png?raw=true" width="400">
</p>

## User Input and Camera Control
Uses cv2 to poll keyboard inputs and pyautogui to poll mouse movements.
These are compiled into an affine transformation that's used to update the camera position and orientation each frame.

# TODO

1. Rendering:
    1. Light transport modelling:
        1. Colour rendering.
        2. Reflections.
        3. Refractions.
    2. Sampling:
        1. Multiple samples per pixel.
2. Control:
    1. Replace the janky cv2/pyautogui combo with something more unified - pygame or something?

# References:
1. Inigo Quilez, https://iquilezles.org/.