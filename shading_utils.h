#ifndef SHADING_UTILS_H
#define SHADING_UTILS_H

#include <glm/glm.hpp>

using namespace glm;

__device__ vec3 reflect(const vec3& v, const vec3& n)
{
	return v - 2 * dot(v, n) * n;
}

#endif
