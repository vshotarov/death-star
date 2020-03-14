#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"

#include <cstdlib>

struct Camera
{
	public:
		__device__ Camera(vec3 look_from, vec3 look_at, vec3 up, float fov, float aspect_ratio)
		// fov is top to bottom in degrees
		{
			vec3 u, v , w;
			float theta = fov * M_PI/180;
			float half_height = tan(theta/2);
			float half_width = aspect_ratio * half_height;

			w = unit_vector(look_from - look_at);
			u = unit_vector(cross(up, w));
			v = cross(w, u);

			origin = look_from;
			lower_left_corner = origin - half_width*u - half_height*v - w;
			horizontal = 2 * half_width * u;
			vertical = 2 * half_height * v;
		}

		__device__ ray get_ray(float u, float v)
		{
			return ray(origin,
				  unit_vector(lower_left_corner + u*horizontal + v*vertical - origin));
		}

	private:
			vec3 lower_left_corner;
			vec3 horizontal;
			vec3 vertical;
			vec3 origin;
};

#endif
