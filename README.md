# Ray Marching
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
I've implemented a rudimentary constructive geometry system for signed distance functions, largely based on the work of Inigo Quilez [1]. This includes: 
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
3. Methods to combine multiple SDFs into new ones:
    1. union, and
    2. smooth union.
 
#### Quaternions and 3D Rotations
Wherever possible, I've encoded rotations with unit quaternions. 

The space of 3D rotations has a ``hole" in it. Paths through the space of 3D rotations can become knotted on this hole, which negatively impacts gradient descent algorithms. For example, consider a sequence $R_0, R_1, ..., R_N$ of rotations where $R_0=R_N=I$ and each $R_i^{-1}R_{i+1}$ is small. If the sequence traces out a path around the hole, it will be impossible to continuously deform it into a path that doesn't, such as the path where each $R_i$ is the identity. This phenomenon doesn't occur if the rotations are encoded with quaternions instead.

Using quaternions everywhere also allows us to track the spin orientation between objects as they move around the scene. Going further, we can design shaders that actually render objects differently based on their spin orientation relative to the camera.

### Ray Marching
#### Sphere Tracing
Given a signed distance function $f$, we can trace a ray from position $p_0$ in direction $v$ using the formula $$p_{i+1} = p_i + f(p_i) \times v.$$ If the ray intersects the surface described by $f$, then $p_i$ will approach the intersection point on the surface.

#### Reflected Ray Directions
Consider a ray starting at $p_0$ and travelling in the direction $v_0$, and let $v_{i+1}$ be the direction of travel after $p_i$. If $p_i$ lies on the surface the outgoing ray will be reflected, otherwise it will continue in the same direction. If $N_i$ is the surface normal at $p_i$, then

$$
v_{i+1} = \begin{cases}
  v_i - 2(v_i\centerdot N_i)N_i & \text{if } f(p_{i}) \approx 0,\\
  v_i  & \text{otherwise.}
\end{cases}
$$

We discuss how to calculate the surface normals $N_i$ from the signed distance function $f$ below.

### Shaders
#### Depth Shader
We can render a depth map by normalising the values of $\log \lvert p_i - p_0 \rvert$ to the interval between $0$ and $1$.
Brighter pixels represent rays that travelled further from the camera.

<p align="center">
  <img src="gallery/distance.png?raw=true" width="600"> 
</p>


#### Surface Proximity Shader
We can render a *surface proximity* map by normalising the values of $\log f(p_i)$ to the interval between $0$ and $1$.
Brighter pixels represent rays that terminated further away from a surface.

<p align="center">
  <img src="gallery/proximity.png?raw=true" width="600"> 
</p>


#### Vignette
Vignetting is a consequence of projecting onto a flat plane, and causes pixels nearer the edge of the image to receive less light than those in the center. This is because:
1. Pixels further to the edge of the image aren't oriented to point directly at the focal point. 
2. Pixels further to the edge of the image are further away from the focal point, so the flux density of incoming photons is reduced by an inverse-square law.

For a pixel $p$ with unit normal $n_p$, and letting $v_p$ be a unit vector from the pixel $p$ to the focal point, we can express the impact of the pixel orientation with a factor of $(v_p \centerdot n_p )$. Let $f$ be the focal distance and $r_p$ is the radial distance between the pixel $p$ and the line of focus. Then the total reduction in the number of incident photons is by a factor of 

$$\frac{f^2}{f^2+r_p^2}(v_p \centerdot n_p)=(\cos \theta)^3=(v_p \centerdot n_p)^3.$$ 

<p align="center">
  <img src="gallery/vignette.png?raw=true" width="600"> 
</p>

#### Surface Normals
Once we've found a ray-surface intersection point $p$, we can calculate the surface normal vector $N_p$ at $p$ by querying the SDF at nearby points and estimating the gradient numerically:
$N_p = (\nabla f)(p)$. 

Concretely, for $i=0, 1, 2, 3$ let $v_i$ be the vertices of a regualar tetrahedron centred at $0$. We estimate the derivative of $f$ in the direction of $v_i - v_0$ using 

$$(\nabla f)(p)\centerdot( \epsilon v_i -  \epsilon v_0) \approx f(p + \epsilon v_i) - f(p + \epsilon v_0),$$

where $\epsilon$ is a small constant. As know the values of $p$, each $v_i$, and each $f(p+\epsilon v_i)$ this gives us $3$ equations in $3$ unknowns at every point $p$, which allows the gradient to be found by solving a simple linear system. As a curiosity, it's also possible to find the surface normal vector by back-propagating the value of $f(p)$ and inspecting the gradient of $p$. This is more commputationally intensive, but also more numerically accurate.

The Surface Normal shader colours pixels based on the coordinates of the surface normal vectors. To generate the image below, I translated surface normals $N_p = (x, y, z)$ to RGB values using $(\lvert x \rvert , \lvert y \rvert, \lvert z \rvert)$.

<p align="center">
  <img src="gallery/normal.png?raw=true" width="600"> 
</p>

#### Surface Laplacian
The Laplacian represents the difference between $f(p)$ and the average value of $f$ over a small sphere centred on $p$. 
We can reuse the values of $f(p+\epsilon v_i)$ that we calculated to find $\nabla f$ to numerically estimate the Laplacian $\Delta f$ of $f$.
Explicitly, for a function $f: \mathbb{R}^n\to\mathbb{R}$ the average value $\overline{f}(p, h)$ of $f$ over a sphere of radius $h$ centred at $p$ is given by
$$\overline{f}(p, h) \approx f(p) - \frac{(\Delta f)(p)}{2n}h^2,$$
where we adopt the positive-definite sign convention for $\Delta$. Then, using 

$$\overline{f}(p, \epsilon)  \approx \frac{1}{4}\sum_{i=0}^3f(p+\epsilon v_i),$$

we get 

$$(\Delta f)(p) \approx \frac{2n}{\epsilon^2} \left( f(p) - \overline{f}(p, \epsilon) \right)$$

$$\approx \frac{6}{\epsilon^2} \left( f(p) - \frac{1}{4}\sum_{i=0}^3f(p+\epsilon v_i) \right).$$


Using the method described above, we calculate the Laplacian $L_p = (\Delta f)(p_i)$ at the terminal location of each pixel's ray, and visualise the resulting scalar field by normalising the values between $0$ and $1$. The image below has also been gamma corrected to improve perceptual uniformity.

<p align="center">
  <img src="gallery/laplacian.png?raw=true" width="600"> 
</p>


#### Lambertian Shader
Very simple geometric illumination model. Intensity $I_p$ is proportional to the cosine between the ray direction $v_p$ and the surface normal $N_p \approx (\nabla f)_p$. Symbolically, $I_p = v_p \centerdot N_p$. 

<p align="center">
  <img src="gallery/lambertian.png?raw=true" width="600"> 
</p>


#### Tangents Shader
Roughly speaking, the Lambertian shader above picks out the component of the surface normals $N_p$ in the direction of the camera. In contrast, the Tangents shader described below picks out the remaining two components.

Let $e_\alpha$ and $e_\beta$ be an orthonormal basis for the sensor plane, and let $v_p$ be a unit vector from the pixel $p$ to the focal point. We project the surface normals $N_p$ onto the plane of the camera sensor,

$$ T_p = N_p - (N_p\centerdot v_p)v_p, $$

and express the resulting vector $T_p$ in terms of $e_\alpha$ and $e_\beta$ as

$$ T_p = \alpha e_\alpha + \beta e_\beta. $$

This gives a vector $(\alpha, \beta)\in\mathbb{R}^2$, and it can be shown that $\alpha^2+\beta^2 \leq 1$. We can treat these projected vectors as complex numbers $\alpha + i\beta$, and apply domain colouring techniques to visualise the result:

<p align="center">
  <img src="gallery/tangent.png?raw=true" width="600"> 
</p>

#### Spin Shader
This shader colours each pixel based on the normal vector of the incident surface and the quaternion that defines the embedding of the camera and object in world space. The result mimics the behaviour of spin-$1/2$ objects in physics, as rotating the camera $360^\circ$ reverses the orientation of the texture on the surface. If $N_p=(x, y, z)$ is the surface normal vector at pixel $p$, and $q$ is the quaternion encoding the embedding of the camera into space, we can define 
$a + bi + cj + dk = \overline{q} N_p \in \mathbb{S}^3$. The next step is to map this value to a colour, which we do by first applying the function 

$$ a + bi + cj + dk \mapsto (a^2 - b^2 - c^2 - d^2) + 2a\sqrt{b^2 + c^2 + d^2} i \in \mathbb{S}^1,$$

and then using a cyclic colourmap to translate points on $\mathbb{S}^1$ to pixel RGB values.

<p align="center">
  <img src="gallery/spin.png?raw=true" width="600">
</p>

Altogether, the sequence of mappings are:

$$ \mathbb{S}^3\times\mathbb{S}^2 \to \mathbb{S}^3 \to \mathbb{S}^1 \to \mathbb{R}^3_{\geq 0}. $$

### User Input and Camera Control
We use Pynput to process mouse and keyboard inputs.

The user can control the camera position and orientation, as well as the settings for a couple of rendering options. Each frame, we need to perform the following steps:
1. Query the mouse and keyboard inputs.
2. Map these to transformations in the parameters they control.
3. Apply those transformations to the relevant parameters.


#### Lie Groups and Algebras
Let $X$ be a set of parameters that transform under the action of a Lie group $G$. The Lie algebra $\mathfrak{g} \cong T_0G$ of $G$ can be used to parameterise small elements of $G$, and provides a convenient encoding for the updates to $X$.

The advantage of working in the Lie algebra $\mathfrak{g}$ is that we can linearly combine the updates using vector addition before mapping the result into $G$ and applying the result to our state parameters. The alternative would involve mapping multiple elements of $\mathfrak{g}$ into $G$ before applying them to the state parameters in sequence, which is more computationally intense and has worse numeric properties.

In our application, $X = G \cong \mathbb{R}^3 \rtimes \text{Spin}_3$ is the set of embeddings into 3D space, and the Lie algebra is $\mathfrak{g} \cong \mathbb{R}^3 \times \mathfrak{sl}_3$. 

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