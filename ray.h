#ifndef RAY_H
#define RAY_H

#include "vec3.h"

struct ray
{
	__device__ ray() {}
	__device__ ray(vec3 origin, vec3 direction) :
		origin(origin), direction(direction) {}

	vec3 origin;
	vec3 direction;

	__device__ vec3 point_at_parameter(float t) const
	{
		return origin + t * direction;
	}
};

#endif
