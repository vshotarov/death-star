#ifndef RAY_H
#define RAY_H

#include <glm/glm.hpp>

using namespace glm;

struct ray
{
	ray(vec3 origin, vec3 direction) :
		origin(origin), direction(direction) {}

	vec3 origin;
	vec3 direction;

	vec3 point_at_parameter(float t) const
	{
		return origin + t * direction;
	}
};

#endif
