#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"
#include "random.h"

struct Camera
{
	public:
		__device__ Camera(vec3 look_from, vec3 look_at, vec3 up, float fov,
				float aspect_ratio, float aperture, float focus_distance)
		// fov is top to bottom in degrees
		{
			lens_radius = aperture / 2;
			float theta = fov * M_PI/180;
			float half_height = tan(theta/2);
			float half_width = aspect_ratio * half_height;

			w = unit_vector(look_from - look_at);
			u = unit_vector(cross(up, w));
			v = cross(w, u);

			origin = look_from;
			lower_left_corner = origin
							  - half_width * focus_distance * u
							  - half_height * focus_distance * v
							  - focus_distance * w;
			horizontal = 2 * half_width * focus_distance * u;
			vertical = 2 * half_height * focus_distance * v;
		}

		__device__ ray get_ray(float s, float t, curandState* rand_state)
		{
			vec3 rd = lens_radius * random_in_unit_disk(rand_state);
            vec3 offset = u * rd.x() + v * rd.y();
            return ray(origin + offset,
                       lower_left_corner + s*horizontal + t*vertical
                           - origin - offset);
		}

	private:
			vec3 lower_left_corner;
			vec3 horizontal;
			vec3 vertical;
			vec3 origin;
			vec3 u, v, w;
			float lens_radius;
};

#endif
