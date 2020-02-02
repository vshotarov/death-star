#ifndef SCENES_H
#define SCENES_H

#include "hittable.h"

#include <curand_kernel.h>


__device__
void create_sphere_and_two_triangles_scene(Hittable** hittables,
		HittableWorld** world)
{
	hittables[0] = Hittable::sphere(vec3(.0, .0, -1.0), .5);
	hittables[1] = Hittable::triangle(vec3(-1.5f, 0.0f, -1.0f),
								  vec3(-2.0f, 1.0f, -2.0f),
								  vec3(-3.0f, 0.0f, -3.0f));
	hittables[2] = Hittable::triangle(vec3(.0f, 0.0f, -1.0f),
								  vec3(-.5f, 0.5f, -1.0f),
								  vec3(-1.5f, -.2f, -1.0f));

	*world = new HittableWorld(hittables, 3);
}

__device__
void create_random_spheres_and_triangles_scene(Hittable** hittables,
		HittableWorld** world, int num_hittables = 50)
{
	curandState rand_state;
	int seed = 2020;
	curand_init(seed, 0, 0, &rand_state);

	for(int i=0; i<num_hittables; i++)
	{
		// half spheres and half triangles
		if(curand_uniform(&rand_state) > .5f)
		{
			hittables[i] = Hittable::sphere(
					4.0f * (vec3(curand_uniform(&rand_state),
						         curand_uniform(&rand_state),
						         curand_uniform(&rand_state) - 1.0f) - .5f),
					curand_uniform(&rand_state));
		}
		else
		{
			hittables[i] = Hittable::triangle(
					4.0f * (vec3(curand_uniform(&rand_state),
						 	     curand_uniform(&rand_state),
						 	     curand_uniform(&rand_state) - 1.0f) - .5f),
					4.0f * (vec3(curand_uniform(&rand_state),
						 	     curand_uniform(&rand_state),
						 	     curand_uniform(&rand_state) - 1.0f) - .5f),
					4.0f * (vec3(curand_uniform(&rand_state),
						 	     curand_uniform(&rand_state),
						 	     curand_uniform(&rand_state) - 1.0f) - .5f));
		}
	}

	*world = new HittableWorld(hittables, num_hittables);
}

#endif
