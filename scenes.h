#ifndef SCENES_H
#define SCENES_H

#include "material.h"

#include <curand_kernel.h>

__device__
void create_RTOW_three_spheres_on_top_of_big_sphere_scene(Hittable* hittables,
		HittableWorld& world)
{
	hittables[0] = Hittable::sphere(vec3(.0, .0, -1.0), .5,
			Material::lambertian(vec3(.8, .3, .3)));
	hittables[1] = Hittable::sphere(vec3(.0, -100.5, -1.0), 100,
			Material::lambertian(vec3(.8, .8, .0)));
	hittables[2] = Hittable::sphere(vec3(1.0, 0.0, -1.0), .5,
			Material::metal(vec3(.8, .6, .2), 0.0f));
	hittables[3] = Hittable::sphere(vec3(-1.0, 0.0, -1.0), .5,
			Material::metal(vec3(.8, .8, .8), 0.3f));

	world = HittableWorld(hittables, 4);
}

__device__
void create_sphere_on_top_of_big_sphere_scene(Hittable* hittables,
		HittableWorld& world)
{
	Material* material = Material::lambertian(vec3(.5, .5, .5));

	hittables[0] = Hittable::sphere(vec3(.0, .0, -1.0), .5, material);
	hittables[1] = Hittable::sphere(vec3(.0, -100.5, -1.0), 100, material);

	world = HittableWorld(hittables, 2);
}

__device__
void create_sphere_and_two_triangles_scene(Hittable* hittables,
		HittableWorld& world)
{
	Material* material = Material::lambertian(vec3(.5, .5, .5));

	hittables[0] = Hittable::sphere(vec3(.0, .0, -1.0), .5, material);
	hittables[1] = Hittable::triangle(vec3(-1.5f, 0.0f, -1.0f),
								  vec3(-2.0f, 1.0f, -2.0f),
								  vec3(-3.0f, 0.0f, -3.0f), material);
	hittables[2] = Hittable::triangle(vec3(.0f, 0.0f, -1.0f),
								  vec3(-.5f, 0.5f, -1.0f),
								  vec3(-1.5f, -.2f, -1.0f), material);

	world = HittableWorld(hittables, 3);
}

__device__
void create_random_spheres_and_triangles_scene(Hittable* hittables,
		HittableWorld& world, int num_hittables = 50)
{
	curandState rand_state;
	int seed = 2020;
	curand_init(seed, 0, 0, &rand_state);

	for(int i=0; i<num_hittables; i++)
	{
		// half spheres and half triangles
		if(curand_uniform(&rand_state) > .5f)
		{
			Material* material = Material::metal(vec3(.5, curand_uniform(&rand_state), .5),
					curand_uniform(&rand_state) * .2f);
			hittables[i] = Hittable::sphere(
					4.0f * (vec3(curand_uniform(&rand_state),
						         curand_uniform(&rand_state),
						         curand_uniform(&rand_state) - 1.0f) - .5f),
					curand_uniform(&rand_state), material);
		}
		else
		{
			Material* material = Material::lambertian(vec3(
						curand_uniform(&rand_state), curand_uniform(&rand_state), .5));
			hittables[i] = Hittable::triangle(
					4.0f * (vec3(curand_uniform(&rand_state),
						 	     curand_uniform(&rand_state),
						 	     curand_uniform(&rand_state) - 1.0f) - .5f),
					4.0f * (vec3(curand_uniform(&rand_state),
						 	     curand_uniform(&rand_state),
						 	     curand_uniform(&rand_state) - 1.0f) - .5f),
					4.0f * (vec3(curand_uniform(&rand_state),
						 	     curand_uniform(&rand_state),
						 	     curand_uniform(&rand_state) - 1.0f) - .5f),
					material);
		}
	}

	world = HittableWorld(hittables, num_hittables);
}

__device__
void create_BVH_test_scene(Hittable* hittables, HittableWorld& world)
{
	Material* material = Material::lambertian(vec3(.5, .5, .5));
	for(int i=0; i<3; i++)
	{
		for(int j=0; j<3; j++)
		{
			hittables[i*3+j] = Hittable::sphere(vec3((i-1)*2,(j-1)*2,-1.0), .5, material);
		}
	}
	world = HittableWorld(hittables, 9);
}

#endif
