# Death Star
A basic cuda path tracer, based on the concepts in Peter Shirley's [Ray Tracing In One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html). The only dependency is [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader) for loading wavefront .obj files.

<img style="padding-bottom: 5px" alt="" src="imgs/08_RTOW_random_spheres_dragon.png?raw=true">

The approach I've taken is more or less identical to what Peter Shirley describes, but I have had to make small adjustments, in order to make the code appropriate for running on the GPU, such as avoiding virtual functions (*and inheritance in general*) and using the [Parallel BVH as proposed by Tero Karras](https://research.nvidia.com/sites/default/files/publications/karras2012hpg_paper.pdf), rather than the recursive one described by Peter Shirley in [Ray Tracing, The Next Week](https://raytracing.github.io/books/RayTracingTheNextWeek.html).

This project was done (*and unfortunately redone many times*) as a part of my ongoing research into parallel computing and general computer graphics. Even though, it has incredibliy limited capabilities, I consider it complete as I feel that going deeper into BRDFs, lights, textures, etc. would be a massive undertaking with a disproportionate reward, so the only addition I might look into in the future is the OpenGL or Vulkan interoperability.

Here is the image that you get as an output if you run the current version of this repo. It is still noisy, as I ran it with 50 samples, a maximum of 25 bounces and a resolution of 1000x500, which took 30m19.357s

<img style="padding-bottom: 5px" alt="" src="imgs/08_Stanford_dragon_and_bunny.png?raw=true">

The following are images I've stored in a sort of chronological order. Prepare to see a lot of spheres

<img style="padding-bottom: 5px" alt="Visualizing normals" src="imgs/01.png?raw=true" width="100" align="left">
<img style="padding-bottom: 5px" alt="Visualizing normals - Random spheres and triangles" src="imgs/02.png?raw=true" width="100" align="left">
<img style="padding-bottom: 5px" alt="Visualizing Ambient Occlusion - Random spheres and triangles" src="imgs/03_basic_AO.png?raw=true" width="100" align="left">
<img style="padding-bottom: 5px" alt="Visualizing Ambient Occlusion - Two spheres" src="imgs/03_basic_AO_sphere.png?raw=true" width="100" align="left">
<img style="padding-bottom: 5px" alt="Metal material" src="imgs/04_metal.png?raw=true" width="100" align="left">
<img style="padding-bottom: 5px" alt="Metal material with colours (albedo)" src="imgs/04_metal_triangles_and_spheres.png?raw=true" width="100" align="left">
<img style="padding-bottom: 5px" alt="Visualizing metal fuzzy reflections" src="imgs/04_RTOW.png?raw=true" width="100">

<p>And some wavefront .obj models (<i>I don't have a good answer as to why did I use stupidly ugly low res sphere and cube that intersect the floor for testing</i>)</p>
<img style="padding-bottom: 5px" alt="Dielectric material and first loaded wavefront .obj files" src="imgs/05_RTOW_dielectric.png?raw=true" width="100" align="left">
<img style="padding-bottom: 5px" alt="Positionable camera" src="imgs/06_RTOW_positionable_camera.png?raw=true" width="100" align="left">
<img style="padding-bottom: 5px" alt="Defocus blur" src="imgs/07_defocus_blur.png?raw=true" width="100">

<p>And then the obligatory "Ray tracing in one weekend" images, with some .obj additions</p>
<img style="padding-bottom: 5px" alt="" src="imgs/08_RTOW_random_spheres_RTOW_camera.png?raw=true">
<img style="padding-bottom: 5px" alt="" src="imgs/08_RTOW_random_spheres.png?raw=true">
<img style="padding-bottom: 5px" alt="" src="imgs/08_RTOW_random_spheres_RTOW_bunny.png?raw=true">
<img style="padding-bottom: 5px" alt="" src="imgs/08_RTOW_random_spheres_random_Y.png?raw=true">

I am not really a Star Wars fan, but for some reason it was the first name I thought of in regards to shooting rays.
