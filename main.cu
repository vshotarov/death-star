#include <cuda.h> // Include so GLM picks up the compiler version
#define GLM_FORCE_CUDA

#include "camera.h"
#include "hittable.h"
#include "scenes.h"
#include "bvh.h"

#include <float.h>
#include <iostream>
#include <fstream>
#include <cstdlib>

#include <curand_kernel.h>
#include <thrust/sort.h>

#include <glm/glm.hpp>

using namespace glm;


__device__
vec3 miss_colour(const ray& r)
{
	vec3 unit_direction = normalize(r.direction);
	float t = .5 * (unit_direction.y + 1.0);
	return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(.5, .7, 1.0);
}

__device__
vec3 colour(const ray& r, HittableWorld* world, curandState* rand_state, int max_bounces)
{
	hit_record rec;
	ray this_ray(r.origin, r.direction);
	vec3 out_colour(0, 0, 0);
	vec3 attenuation(1,1,1);

	for(int d=0; d<max_bounces; d++)
	{
		if(world->hit(this_ray, .0001, MAXFLOAT, rec))
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
void initialize_renderer(int width, int height, curandState* rand_state,
		Camera** camera)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if((x >= width) || (y >= height))
		return;

	int pixel_id = y * width + x;

	// Initialize random number states for each pixel
	int seed = 2020;
	curand_init(seed, pixel_id, 0, &rand_state[pixel_id]);

	// Initialize only one camera
	if(pixel_id == 0)
		*camera = new Camera((float)width / (float)height);
}

__global__
void render(int width, int height, int num_samples, int max_bounces, float* pixel_buffer,
		HittableWorld** world, curandState* rand_state, Camera** camera)
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
		ray r = (*camera)->get_ray(u, v);

		// Ray trace
		out_colour += colour(r, *world, &local_rand_state, max_bounces);
	}

	out_colour /= float(num_samples);

	// Colour space transformation
	out_colour = vec3(sqrt(out_colour.x), sqrt(out_colour.y), sqrt(out_colour.z));

	// Store in pixel buffer
	pixel_buffer[pixel_id * 3 + 0] = out_colour.x;
	pixel_buffer[pixel_id * 3 + 1] = out_colour.y;
	pixel_buffer[pixel_id * 3 + 2] = out_colour.z;
}

__global__
void create_world(Hittable** hittables, HittableWorld** world)
{
	create_BVH_test_scene(hittables, world);
}

__global__
void create_BVH(Hittable** hittables, HittableWorld** world)
	// NOTE: I should probably get rid of the hittable world object
	//	as really, the BVH will take over it's functionality
	// NOTE: This is currently a kernel, so I am not going to be launching
	//	kernels for the subtasks that can be parallelized, in order to simplify.
	// 	Once I am convinced this algorithm works, then I can proceed and change this
	// 	function to a host function that exceutes every parallel task in a separate kernel.
{
	int num_hittables = (*world)->size();

	// Create morton codes for each centroid  NOTE: Do this in parallel
	unsigned int *morton_codes, *sorted_IDs;
	morton_codes = (unsigned int*)malloc(num_hittables * sizeof(unsigned int));
	sorted_IDs = (unsigned int*)malloc(num_hittables * sizeof(unsigned int));

	// We will need to normalize centroids to unit cube [0, 1]
	vec3 scene_size = (*world)->bounding_box.max
					- (*world)->bounding_box.min;

	for(int i=0; i<num_hittables; i++)
	{
		morton_codes[i] = morton3D((hittables[i]->bounding_box->centroid
								 - (*world)->bounding_box.min) / scene_size);
		sorted_IDs[i] = (unsigned int)i;
	}

	// Sort
	// NOTE: This is currently sorting sequentially, but once create_BVH
	// is no longer a kernel, we'll use thrust::device to parallelize it
	thrust::sort_by_key(thrust::seq,
			morton_codes, morton_codes + num_hittables, sorted_IDs);

	// Build tree  NOTE: Do this in parallel
	for(int i=0; i<num_hittables-1; i++)
	{
		// Determine direction of the range
		uint2 range = determine_range(morton_codes, i, num_hittables);
		printf("%i: (%i -> %i)\n", i, range.x, range.y);

		// Find the split position using binary search
		int split_position = find_split(morton_codes, range, i, num_hittables);
		printf("	split %i\n", split_position);

		if(split_position == range.x)
			printf("	left = leaf_nodes[%i]\n", split_position);
		else
			printf("	left = internal_nodes[%i]\n", split_position);

		if(split_position + 1 == range.y)
			printf("	right = leaf_nodes[%i]\n", split_position + 1);
		else
			printf("	right = internal_nodes[%i]\n", split_position + 1);
	}

	// Free memory
	free(morton_codes);
	free(sorted_IDs);
}

int main(int argc, char** argv)
{
	// Parse arguments
	// NOTE this is very erronous at the moment as there is no
	// error catching, validation, etc.
	char *endptr;
	int width = strtol(argv[1], &endptr, 10);
	int height = strtol(argv[2], &endptr, 10);
	int num_samples = strtol(argv[3], &endptr, 10);
	int max_bounces = strtol(argv[4], &endptr, 10);
	char* out_file = argv[5];

	printf("Initializing death-star for %ix%i pixels, %i samples and %i max bounces\n",
			width, height, num_samples, max_bounces);

	// Calculate blocks and threads
	int tx = 8, ty = 8; // bucket size
	
	dim3 blocks(width/tx + 1, height/ty + 1);
	dim3 threads(tx, ty);

	// CUDA random number generator
	curandState *rand_state;
	cudaMalloc((void**)&rand_state, (width * height) * sizeof(curandState));

	// Camera
	Camera** camera;
	cudaMalloc((void**)&camera, 1 * sizeof(Camera));

	initialize_renderer<<<blocks, threads>>>(width, height, rand_state,
			camera);

	// Create scene
	Hittable** hittables;
	HittableWorld** world;

	cudaMalloc((void**)&hittables, 50 * sizeof(Hittable*));
	cudaMalloc((void**)&world, 1 * sizeof(HittableWorld*));

	create_world<<<1, 1>>>(hittables, world);

	// Create BVH
	create_BVH<<<1, 1>>>(hittables, world);

	// Allocate memory for pixels
	float *pixel_buffer, *d_pixel_buffer;
	pixel_buffer = (float*)malloc(width * height * 3 * sizeof(float));
	cudaMalloc(&d_pixel_buffer, width * height * 3 * sizeof(float));

	// Render into buffer
	render<<<blocks, threads>>>(width, height, num_samples, max_bounces, d_pixel_buffer,
			world, rand_state, camera);

	// Copy pixel data from device to cpu
	cudaMemcpy(pixel_buffer, d_pixel_buffer,
			width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost);

	// Write into ppm file
	std::ofstream out(out_file);
	std::streambuf *coutbuf = std::cout.rdbuf(); // Store old buf
	std::cout.rdbuf(out.rdbuf()); // Redirect cout to out_file

	std::cout<< "P3\n" << width << " " << height << "\n255\n";
	for(int y=height-1; y>=0; y--)
		for(int x=0; x<width; x++)
		{
			int pixel_id = y * width + x;

			int int_r = int(255.99 * pixel_buffer[pixel_id * 3 + 0]);
			int int_g = int(255.99 * pixel_buffer[pixel_id * 3 + 1]);
			int int_b = int(255.99 * pixel_buffer[pixel_id * 3 + 2]);

			std::cout<< int_r <<" "<< int_g << " " << int_b <<std::endl;
		}

	// Restore cout buf
	std::cout.rdbuf(coutbuf);

	return 0;
}
