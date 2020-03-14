#ifndef SCENES_H
#define SCENES_H

#include "material.h"
#include "hittable.h"
#include "scene.h"

#include <curand_kernel.h>
#include <iostream>


__device__
void create_RTOW_three_spheres_on_top_of_big_sphere_scene(Hittable* hittables, int start_id)
{
	hittables[start_id + 0] = Hittable::sphere(vec3(.0, .0, -1.0), .5,
			Material::lambertian(vec3(.8, .3, .3)));
	hittables[start_id + 1] = Hittable::sphere(vec3(.0, -100.5, -1.0), 100,
			Material::lambertian(vec3(.8, .8, .0)));
	hittables[start_id + 2] = Hittable::sphere(vec3(1.0, 0.0, -1.0), .5,
			Material::metal(vec3(.8, .6, .2), 0.0f));
	hittables[start_id + 3] = Hittable::sphere(vec3(-1.0, 0.0, -1.0), .5,
			Material::metal(vec3(.8, .8, .8), 0.3f));
}

__device__
void create_RTOW_glass_sphere(Hittable* hittables, int start_id)
{
	hittables[start_id + 0] = Hittable::sphere(vec3(.0, .0, -1.0), .5,
			Material::lambertian(vec3(.1, .2, .5)));
	hittables[start_id + 1] = Hittable::sphere(vec3(.0, -100.5, -1.0), 100,
			Material::lambertian(vec3(.8, .8, .0)));
	hittables[start_id + 2] = Hittable::sphere(vec3(1.0, 0.0, -1.0), .5,
			Material::metal(vec3(.8, .6, .2), 0.0f));
	hittables[start_id + 3] = Hittable::sphere(vec3(-1.0, 0.0, -1.0), .5,
			Material::dielectric(1.5));
	hittables[start_id + 4] = Hittable::sphere(vec3(-1.0, 0.0, -1.0), -.45,
			Material::dielectric(1.5));
}

__device__
void create_sphere_on_top_of_big_sphere_scene(Hittable* hittables, int start_id)
{
	Material* material = Material::lambertian(vec3(.5, .5, .5));

	hittables[start_id + 0] = Hittable::sphere(vec3(.0, .0, -1.0), .5, material);
	hittables[start_id + 1] = Hittable::sphere(vec3(.0, -100.5, -1.0), 100, material);
}

__device__
void create_sphere_and_two_triangles_scene(Hittable* hittables, int start_id)
{
	Material* material = Material::lambertian(vec3(.5, .5, .5));

	hittables[start_id + 0] = Hittable::sphere(vec3(.0, .0, -1.0), .5, material);
	hittables[start_id + 1] = Hittable::triangle(vec3(-1.5f, 0.0f, -1.0f),
								  vec3(-2.0f, 1.0f, -2.0f),
								  vec3(-3.0f, 0.0f, -3.0f), material);
	hittables[start_id + 2] = Hittable::triangle(vec3(.0f, 0.0f, -1.0f),
								  vec3(-.5f, 0.5f, -1.0f),
								  vec3(-1.5f, -.2f, -1.0f), material);
}

__device__
void create_random_spheres_and_triangles_scene(Hittable* hittables, int start_id, int num_hittables = 50)
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
			hittables[start_id + i] = Hittable::sphere(
					4.0f * (vec3(curand_uniform(&rand_state),
						         curand_uniform(&rand_state),
						         curand_uniform(&rand_state) - 1.0f) - .5f),
					curand_uniform(&rand_state), material);
		}
		else
		{
			Material* material = Material::lambertian(vec3(
						curand_uniform(&rand_state), curand_uniform(&rand_state), .5));
			hittables[start_id + i] = Hittable::triangle(
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
}

__device__
void create_BVH_test_scene(Hittable* hittables, int start_id)
{
	Material* material = Material::lambertian(vec3(.5, .5, .5));
	for(int i=0; i<3; i++)
	{
		for(int j=0; j<3; j++)
		{
			hittables[start_id + i*3+j] = Hittable::sphere(vec3((i-1)*2,(j-1)*2,-1.0), .5, material);
		}
	}
}

__device__
void create_RTOW_random_spheres_scene(Hittable* hittables, int start_id, curandState* rand_state)
{
    hittables[start_id] = Hittable::sphere(vec3(0,-1000,0), 1000, Material::lambertian(vec3(0.5, 0.5, 0.5)));
    int i = start_id + 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = curand_uniform(rand_state);
            vec3 center(a+0.9*curand_uniform(rand_state),0.2,b+0.9*curand_uniform(rand_state));
            if ((center-vec3(4,0.2,0)).length() > 0.9) {
                if (choose_mat < 0.8) {  // diffuse
                    hittables[i++] = Hittable::sphere(center, 0.2,
                        Material::lambertian(vec3(curand_uniform(rand_state)*curand_uniform(rand_state),
                                            curand_uniform(rand_state)*curand_uniform(rand_state),
                                            curand_uniform(rand_state)*curand_uniform(rand_state))
                        )
                    );
                }
                else if (choose_mat < 0.95) { // metal
                    hittables[i++] = Hittable::sphere(center, 0.2,
                            Material::metal(vec3(0.5*(1 + curand_uniform(rand_state)),
                                           0.5*(1 + curand_uniform(rand_state)),
                                           0.5*(1 + curand_uniform(rand_state))),
                                      0.5*curand_uniform(rand_state)));
                }
                else {  // glass
                    hittables[i++] = Hittable::sphere(center, 0.2, Material::dielectric(1.5));
                }
            }
        }
    }

    hittables[i++] = Hittable::sphere(vec3(0, 1, 0), 1.0, Material::dielectric(1.5));
    hittables[i++] = Hittable::sphere(vec3(-4, 1, 0), 1.0, Material::lambertian(vec3(0.4, 0.2, 0.1)));
    hittables[i++] = Hittable::sphere(vec3(4, 1, 0), 1.0, Material::metal(vec3(0.7, 0.6, 0.5), 0.0));
}

#endif
