# Ray Marching
<p align="center">
  <img src="gallery/lambertian.png?raw=true" width="200"> 
  <img src="gallery/normal.png?raw=true" width="200"> 
  <img src="gallery/tangent.png?raw=true" width="200"> 
  <img src="gallery/spin.png?raw=true" width="200">
</p>

## Introduction
*Ray marching* is a method for rendering computer graphics.
In contrast with other rendering methods that use textured meshes, ray marching algorithms operate on a *signed distance field* (SDF) representation of the scene. This repository contains the following: 
1. A constructive geometry system for building complex SDFs by composing simple primitives.
2. Ray marching operations for rendering SDFs.
3. A collection of geometrically motivated shaders.
4. An interactive first person control system for the camera that uses mouse and keyboard inputs.

### Differentiability
All scene construction and rendering functions are implemented in PyTorch and are fully differentiable. This allows us to backproagate gradients through the rendering process, and optimise the scene parameters based on the pixel values of the rendered output image.

## Design and Implementation Details
### SDF Constructive Geometry System
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
 
#### Quaternions and 3D Rotations
Wherever possible, I've encoded rotations with quaternions. 

The space of 3D rotations has a ``hole" in it. Paths through the space of 3D rotations can become knotted on this hole, which negatively impacts gradient descent algorithms. For example, consider a sequence $R_0, R_1, ..., R_N$ of rotations where $R_0=R_N=I$ and each $R_i^{-1}R_{i+1}$ is small. If the sequence traces out a path around the hole, it will be impossible to continuously deform it into a path that doesn't, such as the path where each $R_i$ is the identity. This phenomenon doesn't occur if the rotations are encoded with quaternions instead.

Using quaternions everywhere also allows us to track the spin orientation between objects as they move around the scene. Going further, we can design shaders that actually render objects differently based on their spin orientation relative to the camera.

### Ray Tracing
#### Walk on Spheres
Given a signed distance function $f$, we can trace a ray from position $x_0$ in direction $v_0$ using the formula $x_{i+1} = x_i + f(x_i) \times v_0$. If a ray starting at $x_0$ and travelling in the direction $v_0$ intersects the surface described by $f$, then $x_i$ will approach the intersection point on the surface.

#### Surface Normal Calculation
Once we've found a ray-surface intersection point $p$, we can calculate the surface normal vector at $p$ by querying the SDF at nearby points and estimating the gradient numerically. Alternatively, it's possible to find the surface normal vector by back-propagating the value of $f(p)$ and inspecting the gradient of $p$.

### Shaders
#### Lambertian Shader
Very simple geometric illumination model.
<p align="center">
  <img src="gallery/lambertian.png?raw=true" width="400"> 
</p>

#### Normals Shader
Surfaces are coloured based on the coordinates of their normal vectors. To generate the image below, if $(x, y, z)\in\mathbb{S}^2$ is the normal vector of the surface, we colour it with RGB values $(x^2, y^2, z^2)$.
<p align="center">
  <img src="gallery/normal.png?raw=true" width="400"> 
</p>

#### Tangents Shader
Surface normals are projected onto the plane of the camera sensor, giving a vector $(u, v)\in\mathbb{R}^2$ with $u^2+v^2 \leq 1$. We can treat these projected vectors as complex numbers, and apply domain colouring techniques to visualise the result.
<p align="center">
  <img src="gallery/tangent.png?raw=true" width="400"> 
</p>

#### Spin Shader
Surfaces are coloured based on a combination of their normal vector and the quaternions that define the embeddings of the camera and object in world space. This shader mimics the behaviour of spin-$1/2$ objects in physics --- rotating the camera $360$ degrees reverses the orientation of the texture on the surface.
<p align="center">
  <img src="gallery/spin.png?raw=true" width="400">
</p>

### User Input and Camera Control
Uses cv2 to poll keyboard inputs and pyautogui to poll mouse movements.
These are compiled into an affine transformation that's used to update the camera position and orientation each frame.

### Display
I've used the TorchWindow package [2] to display rendered frames without moving any data off the GPU. 

<!-- #### Lie Groups and Algebras -->

## TODO
1. Rendering:
    1. Light transport modelling:
        1. Colour rendering.
        2. Reflections.
        3. Refractions.
    2. Sampling:
        1. Multiple samples per pixel.
2. Control:
    1. Replace the janky cv2/pyautogui combo with Pynput [3].
3. Speed optimisations:
    1. Support for half precision computations. 
        1. I've played with this but there wasn't a massive speed-up, and it required introducing a new "dtype" keyword that runs through all the classes. I think there's got to be a better approach.
    2. Some kind of bounding box hierarchy implementation?

## References:
1. Inigo Quilez, https://iquilezles.org/.
2. TorchWindow, https://github.com/jbaron34/torchwindow/.
3. Pynput, https://pynput.readthedocs.io/en/latest/index.html.