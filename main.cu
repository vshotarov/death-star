#include <cuda.h> // Include so GLM picks up the compiler version
#define GLM_FORCE_CUDA

#include "ray.h"
#include "hittable.h"

#include <float.h>
#include <iostream>
#include <fstream>
#include <cstdlib>

#include <curand_kernel.h>

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
vec3 colour(const ray& r, HittableWorld* world)
{
	hit_record rec;

	if(world->hit(r, .0001, MAXFLOAT, rec))
	{
		return .5f * (rec.normal + 1.0f);
	}

	return miss_colour(r);
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
void render(int width, int height, int num_samples, float* pixel_buffer,
		HittableWorld** world, curandState* rand_state)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if((x >= width) || (y >= height))
		return;

	int pixel_id = y * width + x;

	// NOTE temporarily defining camera variables here before actually
	// implementing camera
	float aspect_ratio = (float)width / float(height);
	vec3 lower_left_corner(-2.0, -1.0, -1.0);
	vec3 horizontal(2.0 * aspect_ratio, .0, .0);
	vec3 vertical(.0, 2.0, .0);
	vec3 origin(.0, .0, .0);

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
		ray r(origin,
			  normalize(lower_left_corner + u*horizontal + v*vertical));

		// Ray trace
		out_colour += colour(r, *world);
	}

	out_colour /= float(num_samples);

	// Store in pixel buffer
	pixel_buffer[pixel_id * 3 + 0] = out_colour.x;
	pixel_buffer[pixel_id * 3 + 1] = out_colour.y;
	pixel_buffer[pixel_id * 3 + 2] = out_colour.z;
}

__global__
void create_world(Hittable** hittables, HittableWorld** world)
{
	hittables[0] = Hittable::sphere(vec3(.0, .0, -1.0), .5);
	hittables[1] = Hittable::triangle(vec3(-1.5f, 0.0f, -1.0f),
								  vec3(-2.0f, 1.0f, -2.0f),
								  vec3(-3.0f, 0.0f, -3.0f));
	hittables[2] = Hittable::triangle(vec3(.0f, 0.0f, -1.0f),
								  vec3(-.5f, 0.5f, -1.0f),
								  vec3(-1.5f, -.2f, -1.0f));

	*world = new HittableWorld(hittables, 3);
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
	char* out_file = argv[4];

	printf("Initializing death-star for %ix%i pixels and %i samples\n",
			width, height, num_samples);

	// Calculate blocks and threads
	int tx = 8, ty = 8; // bucket size
	
	dim3 blocks(width/tx + 1, height/ty + 1);
	dim3 threads(tx, ty);

	// CUDA random number generator
	curandState *rand_state;
	cudaMalloc((void**)&rand_state, (width * height) * sizeof(curandState));

	initialize_renderer<<<blocks, threads>>>(width, height, rand_state);

	// Create scene
	Hittable** hittables;
	HittableWorld** world;

	cudaMalloc((void**)&hittables, 3 * sizeof(Hittable*));
	cudaMalloc((void**)&world, 1 * sizeof(HittableWorld*));

	create_world<<<1, 1>>>(hittables, world);

	// Allocate memory for pixels
	float *pixel_buffer, *d_pixel_buffer;
	pixel_buffer = (float*)malloc(width * height * 3 * sizeof(float));
	cudaMalloc(&d_pixel_buffer, width * height * 3 * sizeof(float));

	// Render into buffer
	render<<<blocks, threads>>>(width, height, num_samples, d_pixel_buffer,
			world, rand_state);

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
