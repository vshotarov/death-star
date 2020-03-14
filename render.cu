#include "camera.h"
#include "hittable.h"
#include "bvh.h"

#include <float.h>

#include <curand_kernel.h>


__device__
vec3 miss_colour(const ray& r)
{
	vec3 unit_direction = unit_vector(r.direction);
	float t = .5 * (unit_direction.y() + 1.0);
	return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(.5, .7, 1.0);
}

__device__
vec3 colour(const ray& r, BVHNode* bvh_root, curandState* rand_state, int max_bounces)
{
	hit_record rec;
	ray this_ray(r.origin, r.direction);
	vec3 out_colour(0, 0, 0);
	vec3 attenuation(1,1,1);

	for(int d=0; d<max_bounces; d++)
	{
		if(hit_BVH(bvh_root, this_ray, .0001, MAXFLOAT, rec))
		{
			ray scattered;
			vec3 this_attenuation;

			if(rec.material->scatter(this_ray, rec, this_attenuation, scattered, rand_state))
			{
				this_ray = scattered;
				attenuation *= this_attenuation;
			}
			else
			{
				break;
			}
		}
		else
		{
			out_colour = miss_colour(r);
			break;
		}
	}

	return attenuation * out_colour;
}

__global__
void initialize_renderer(int width, int height, curandState* rand_state)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if((x >= width) || (y >= height))
		return;

	int pixel_id = y * width + x;

	// Initialize random number states for each pixel
	int seed = 2020;
	curand_init(seed, pixel_id, 0, &rand_state[pixel_id]);
}

__global__
void initialize_camera(Camera* camera, vec3 look_from, vec3 look_at, vec3 up,
		float fov, float aspect_ratio, float aperture, float focus_distance)
{
	(*camera) = Camera(look_from, look_at, up, fov, aspect_ratio, aperture, focus_distance);
}

__global__
void render(int width, int height, int num_samples, int max_bounces, float* pixel_buffer,
		BVHNode* bvh_root, curandState* rand_state, Camera* camera)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if((x >= width) || (y >= height))
		return;

	int pixel_id = y * width + x;

	float f_width = (float)width;
	float f_height = (float)height;

	// Grab the rand_state for this pixel
	curandState local_rand_state = rand_state[pixel_id];

	vec3 out_colour(.0, .0, .0);

	for(int s=0; s<num_samples; s++)
	{
		// U and V are the 2d coordinates of the camera plane
		float u = float(x + curand_uniform(&local_rand_state)) / f_width;
		float v = float(y + curand_uniform(&local_rand_state)) / f_height;

		// Get ray through the pixel
		ray r = camera->get_ray(u, v, &local_rand_state);

		// Ray trace
		out_colour += colour(r, bvh_root, &local_rand_state, max_bounces);
	}

	out_colour /= float(num_samples);

	// Colour space transformation
	out_colour = vec3(sqrt(out_colour.x()), sqrt(out_colour.y()), sqrt(out_colour.z()));

	// Store in pixel buffer
	pixel_buffer[pixel_id * 3 + 0] = out_colour.x();
	pixel_buffer[pixel_id * 3 + 1] = out_colour.y();
	pixel_buffer[pixel_id * 3 + 2] = out_colour.z();
}
