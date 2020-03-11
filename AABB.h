#ifndef AABB_H
#define AABB_H

#include "hittable.h"

#include <glm/glm.hpp>

using namespace glm;


__device__ inline float ffmin(float a, float b) { return a < b ? a : b; }
__device__ inline float ffmax(float a, float b) { return a > b ? a : b; }

struct AABB
{
	public:
		__device__ AABB() {}
		__device__ AABB(vec3 min, vec3 max) :
			min(min), max(max) { centroid = (min + max) * .5f; }

		vec3 min;
		vec3 max;
		vec3 centroid;
};

__device__ AABB surrounding_box(AABB box0, AABB box1)
{
	vec3 small(ffmin(box0.min.x, box1.min.x),
			   ffmin(box0.min.y, box1.min.y),
			   ffmin(box0.min.z, box1.min.z));
	vec3 big(ffmax(box0.max.x, box1.max.x),
			 ffmax(box0.max.y, box1.max.y),
			 ffmax(box0.max.z, box1.max.z));
	return AABB(small, big);
}

#endif
