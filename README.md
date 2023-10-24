# Ray Marching
<!-- <p align="center">
  <img src="gallery/distance.png?raw=true" width="200"> 
  <img src="gallery/lambertian.png?raw=true" width="200"> 
  <img src="gallery/normal.png?raw=true" width="200"> 
  <img src="gallery/tangent.png?raw=true" width="200">
  <img src="gallery/proximity.png?raw=true" width="200"> 
  <img src="gallery/vignette.png?raw=true" width="200"> 
  <img src="gallery/laplacian.png?raw=true" width="200"> 
  <img src="gallery/spin.png?raw=true" width="200">
</p> -->

<p align="center">
  <img src="gallery/distance.png?raw=true" width="400"> 
  <img src="gallery/proximity.png?raw=true" width="400"> 
  <img src="gallery/lambertian.png?raw=true" width="400"> 
  <img src="gallery/vignette.png?raw=true" width="400"> 
  <img src="gallery/normal.png?raw=true" width="400"> 
  <img src="gallery/laplacian.png?raw=true" width="400"> 
  <img src="gallery/tangent.png?raw=true" width="400">
  <img src="gallery/spin.png?raw=true" width="400">
</p>

## Introduction
<!-- This repository was created as a way for me to experiment with a variety of programming and mathematics concepts, inlcuding:
1. 3D rendering and shading.
2. Computational geometry.
3. Quaternions and their relationship with 3D rotations.
 -->

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
Wherever possible, I've encoded rotations with unit quaternions. 

<!-- The group of unit quaternions double covers the group of rotations in 3D. Let $\phi:\mathbb{S}^3 \to \text{SO}_3$.  -->
<!-- We parameterise affine embeddings in space using the group $\mathbb{S}^3\times\mathbb{R}^3$ using  $p^{(q, t)} = p^{\phi(q)} + t.$ -->

The space of 3D rotations has a ``hole" in it. Paths through the space of 3D rotations can become knotted on this hole, which negatively impacts gradient descent algorithms. For example, consider a sequence $R_0, R_1, ..., R_N$ of rotations where $R_0=R_N=I$ and each $R_i^{-1}R_{i+1}$ is small. If the sequence traces out a path around the hole, it will be impossible to continuously deform it into a path that doesn't, such as the path where each $R_i$ is the identity. This phenomenon doesn't occur if the rotations are encoded with quaternions instead.

Using quaternions everywhere also allows us to track the spin orientation between objects as they move around the scene. Going further, we can design shaders that actually render objects differently based on their spin orientation relative to the camera.

### Ray Tracing
#### Walk on Spheres
Given a signed distance function $f$, we can trace a ray from position $p_0$ in direction $v$ using the formula $$p_{i+1} = p_i + f(p_i) \times v.$$ If a ray starting at $p_0$ and travelling in the direction $v$ intersects the surface described by $f$, then $p_i$ will approach the intersection point on the surface.



<!-- $$(\Delta f)(p) \approx f(p) - \frac{1}{4}\sum_{i=0}^3f(p+\epsilon v_i).$$ -->
<!-- The Laplacian is positive where the surface extrudes outwards, and negative where it curves inwards. -->


### Shaders
#### Depth Shader
Brighter pixels represent rays that travelled further from the camera.
<p align="center">
  <img src="gallery/distance.png?raw=true" width="600"> 
</p>


#### Proximity Shader
Brighter pixels represent rays that terminated further away from a surface.
<p align="center">
  <img src="gallery/proximity.png?raw=true" width="600"> 
</p>


#### Vignette
Vignetting is a consequence of projecting onto a flat plane, and causes pixels nearer the edge of the image to recieve less light than those in the center. This is because:
1. Pixels further to the edge of the image aren't oriented to point directly at the focal point. 
2. Pixels further to the edge of the image are further away from the focal point, so the flux density of incoming photons is reduced by an inverse-square law.

For a pixel $p$ with unit normal $n_p$, and letting $v_p$ be a unit vector from the pixel $p$ to the focal point, we can express the impact of the pixel orientation with a factor of $(v_p \centerdot n_p )$. Let $f$ be the focal distance and $r_p$ is the radial distance between the pixel $p$ and the line of focus. Then the total reduction in the number of incident photons is by a factor of $$\frac{f^2}{f^2+r_p^2}(v_p \centerdot n_p)=(\cos \theta)^3=(v_p \centerdot n_p)^3.$$ 
<p align="center">
  <img src="gallery/vignette.png?raw=true" width="600"> 
</p>




<!-- If $n_p$ is the unit normal vector of the pixel, and $v_p$ is a unit vector pointing from the pixel centre to the focal point, then the intensity $I_p$ at $p$ will be scaled by 

$$I_p = (v_p \centerdot n_p)$$
$$ = (\cos \theta)^2(\cos \theta)$$ -->

#### Normals Shader
Once we've found a ray-surface intersection point $p$, we can calculate the surface normal vector $N_p$ at $p$ by querying the SDF at nearby points and estimating the gradient numerically:
$N_p = (\nabla f)(p)$. 

Concretely, for $i=0, 1, 2, 3$ let $v_i$ be the vertices of a regualar tetrahedron centred at $0$. We estimate the derivative of $f$ in the direction of $v_i - v_0$ using $$(\nabla f)(p)\centerdot( \epsilon v_i -  \epsilon v_0) \approx f(p + \epsilon v_i) - f(p + \epsilon v_0),$$
where $\epsilon$ is a small constant. We know the values of $p$, each $v_i$, and each $f(p+\epsilon v_i)$. This gives us $3$ equations in $3$ unknowns at every point $p$, allowing the gradient to be found by solving a simple linear system. As a curiosity, it's also possible to find the surface normal vector by back-propagating the value of $f(p)$ and inspecting the gradient of $p$. This is more commputationally intensive, but also more numerically accurate.

Surfaces are coloured based on the coordinates of their normal vectors. To generate the image below, I translated surface normals $n_p = (x, y, z)$ to RGB values using $(\lvert x \rvert , \lvert y \rvert, \lvert z \rvert)$.
<p align="center">
  <img src="gallery/normal.png?raw=true" width="600"> 
</p>

#### Surface SDF Laplacian
We can also use the values of $f(p+\epsilon v_i)$ to numerically estimate the Laplacian $\Delta f$ of $f$. This is because the Laplacian represents the difference between $f(p)$ and the average value of $f$ over a small sphere centred on $p$. Explicitly, for a function $f: \mathbb{R}^n\to\mathbb{R}$ the average value $\overline{f}(p, h)$ of $f$ over a sphere of radius $h$ centred at $p$ is given by
$$ \overline{f}(p, h) = f(p) - \frac{(\Delta f)(p)}{2n}h^2 + (\ldots).$$

Using $\overline{f}(p, \epsilon) =  \frac{1}{4}\sum_{i=0}^3f(p+\epsilon v_i)$ we get 
$$(\Delta f)(p) \approx \frac{2n}{h^2} \left( f(p) - \overline{f}(p, \epsilon) \right)$$
$$= \frac{2n}{h^2} \left( f(p) - \frac{1}{4}\sum_{i=0}^3f(p+\epsilon v_i) \right)$$
$$= \frac{6}{h^2} \left( f(p) - \frac{1}{4}\sum_{i=0}^3f(p+\epsilon v_i) \right).$$

<p align="center">
  <img src="gallery/laplacian.png?raw=true" width="600"> 
</p>



#### Lambertian Shader
Very simple geometric illumination model. Intensity $I_p$ is proportional to the cosine between the ray direction $v_p$ and the surface normal $(\nabla f)_p$. Symbolically, $I_p = v_p \centerdot (\nabla f)_p$. 
<p align="center">
  <img src="gallery/lambertian.png?raw=true" width="600"> 
</p>


#### Tangents Shader
Surface normals are projected onto the plane of the camera sensor, giving a vector $(u, v)\in\mathbb{R}^2$ with $u^2+v^2 \leq 1$. We can treat these projected vectors as complex numbers $u + iv$, and apply domain colouring techniques to visualise the result. 

<p align="center">
  <img src="gallery/tangent.png?raw=true" width="600"> 
</p>

#### Spin Shader
Surfaces are coloured based on a combination of their normal vector and the quaternions that define the embeddings of the camera and object in world space. This shader mimics the behaviour of spin-$1/2$ objects in physics, as rotating the camera $360^\circ$ reverses the orientation of the texture on the surface.
$$ (q_0v) / (q_1n) $$
<p align="center">
  <img src="gallery/spin.png?raw=true" width="600">
</p>

### User Input and Camera Control
Uses Pynput to process the user mouse and keyboard inputs.
These are compiled into an affine transformation that's used to update the camera position and orientation each frame. 

The user can control the camera position and orientation, as well as the settings for a couple of rendering options. Each frame, we need to perform the following steps:
1. Query the mouse and keyboard inputs.
2. Map these to transformations in the parameters they control.
3. Apply those transformations to the relevant parameters.


#### Lie Groups and Algebras
$\exp: \mathbb{R}^3 \times \mathfrak{sl}_3 \to \mathbb{R}^3 \rtimes \text{Spin}_3$

### Display
I've used the TorchWindow package [2] to display rendered frames without moving any data off the GPU. 


## TODO
1. Replace the rigid $1$ sample-per-pixel rendering method with a more standard ray tracing method that randomly samples points over screen space, traces rays through them, and then aggregates the values onto pixels.
2. Clean up:
    1. User input handling in control.py
    2. Shader functions in shader.py
    3. The pandas dependency is definitely unnecessary and I should get rid of it.
    4. I'd like to get rid of the pyautogui dependency too, which I'm using to get the scren size.
3. Rendering:
    1. Light transport modelling:
        1. Colour rendering.
        2. Reflections.
        3. Refractions.
    2. Improved sampling:
        1. Dynamic sampling of screen space.
        2. Interpolate samples to colour missing pixels.
4. Control:
    1. Finish removing pyautogui.
        1. Need to query screen size somehow.
    2. Implement ``cyclindical'' movement system with fixed up direction.
    3. Hide mouse cursor.
5. Speed optimisations:
    1. Support for half precision computations. 
        1. I've played with this but it required introducing a new "dtype" keyword that runs through all the classes. I think there's got to be a better approach.
        2. Rendering in half precision causes graphical artifacts on some SDFs, need to investigate how to improve this.
    2. Some kind of bounding box hierarchy implementation?
    3. Add PyTorch profiling script.
    4. Test speed of different implementations of inverse affine transformations.


## References
1. Inigo Quilez, https://iquilezles.org/.
2. TorchWindow, https://github.com/jbaron34/torchwindow/.
3. Pynput, https://pynput.readthedocs.io/en/latest/index.html.