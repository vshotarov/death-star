#include <cuda.h> // Include so GLM picks up the compiler version
#define GLM_FORCE_CUDA

#include "camera.h"
#include "hittable.h"
#include "template_scenes.h"
#include "bvh.h"
#include "loadOBJ.h"
#include "render.cu"

#include <iostream>
#include <fstream>
#include <cstdlib>

#include <curand_kernel.h>

#include <glm/glm.hpp>

using namespace glm;


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
	char* obj_file = NULL;
	if(argc > 6)
		obj_file = argv[6];

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
	Camera* camera;
	cudaMalloc(&camera, 1 * sizeof(Camera));

	initialize_renderer<<<blocks, threads>>>(width, height, rand_state,
			camera);

	// Create scene
	Scene scene;
	scene.num_hittables = 0;
	if(obj_file != NULL)
	{
		load_obj(scene, obj_file);
	}
	else
		create_template_scene(scene);

	// Create BVH
	BVHNode* bvh_root = create_BVH(scene.hittables, scene.num_hittables);

	// Allocate memory for pixels
	float *pixel_buffer, *d_pixel_buffer;
	pixel_buffer = (float*)malloc(width * height * 3 * sizeof(float));
	cudaMalloc(&d_pixel_buffer, width * height * 3 * sizeof(float));

	// Render into buffer
	render<<<blocks, threads>>>(width, height, num_samples, max_bounces, d_pixel_buffer,
			bvh_root, rand_state, camera);

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
