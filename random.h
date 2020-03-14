#ifndef RANDOM_H
#define RANDOM_H

#include <curand_kernel.h>


__device__ vec3 random_in_unit_sphere(curandState* rand_state)
{
	vec3 p;
	do {
		p = 2.0f *  vec3(curand_uniform(rand_state),
						 curand_uniform(rand_state),
						 curand_uniform(rand_state)) - vec3(1,1,1);
	} while (p.x() * p.x() + p.y() * p.y() + p.z() * p.z() >= 1.0);

	return p;
}

__device__ vec3 random_in_unit_disk(curandState* rand_state)
{
	vec3 p;
    do {
        p = 2.0*vec3(curand_uniform(rand_state),curand_uniform(rand_state),0) - vec3(1,1,0);
    } while (dot(p,p) >= 1.0);
    return p;
}

#endif
