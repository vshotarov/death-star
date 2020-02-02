#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"

#include <glm/glm.hpp>

using namespace glm;

struct Camera
{
	public:
		__device__ Camera(float aspect_ratio)
		{
			lower_left_corner = vec3(-2.0, -1.0, -1.0);
			horizontal = vec3(2.0 * aspect_ratio, .0, .0);
			vertical = vec3(.0, 2.0, .0);
			origin = vec3(.0, .0, .0);
		}

		__device__ ray get_ray(float u, float v)
		{
			return ray(origin,
				  normalize(lower_left_corner + u*horizontal + v*vertical));
		}

	private:
			vec3 lower_left_corner;
			vec3 horizontal;
			vec3 vertical;
			vec3 origin;
};

#endif
