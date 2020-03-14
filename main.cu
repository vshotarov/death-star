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


// Since memory for hittables must already be allocated when creating
// them on the GPU, I currently store a static number of how many hittables
// are manually created.
// It is far less than ideal, and a potential workaround would be to instead
// of creating them directly on the GPU, I create a bunch of sphereData and
// triangleData structs, similar to the ones I create for .obj files, which
// are stored on the CPU, so we can use their count to allocate the correct
// amount of memory for hittables, before sending them off the the GPU to be created.
//
// The reason I am not fully keen on that, is that we'll have an extra step
// and also a copy for each sphere and triangle on the CPU, which seems wasteful
//
// Additionally, most high profile renderers, have their own file formats (
// Arnold .ass, Renderman RIB, etc.) that describe a scene, which contain
// the number of objects to render, so in those cases, the number of hittables
// is always known, so there is no need for neither the above mentioned proceedure
// nor the following manually maintained static value.
#define num_manually_defined_hittables 2
__global__
void populate_scene(Hittable* hittables, int start_id)
{
	create_sphere_on_top_of_big_sphere_scene(hittables, start_id);
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

	objData obj = load_obj(obj_file);
	objData obj2 = load_obj("/home/vshotarov/Downloads/bunny.obj");
	scene.num_hittables = obj.num_triangles + obj2.num_triangles + num_manually_defined_hittables;

	cudaMalloc(&(scene.hittables), scene.num_hittables * sizeof(Hittable));

	Material* material;
	cudaMalloc(&(material), sizeof(Material));

	initialize_obj_material<<<1, 1>>>(material);

	int obj_threads = 512;
    int obj_dims = (obj.num_triangles + obj_threads - 1) / obj_threads;
	create_obj_hittables<<<obj_dims, obj_threads>>>(scene.hittables, material, obj, 0);

    obj_dims = (obj2.num_triangles + obj_threads - 1) / obj_threads;
	create_obj_hittables<<<obj_dims, obj_threads>>>(scene.hittables, material, obj2, obj.num_triangles);

	populate_scene<<<1, 1>>>(scene.hittables, obj.num_triangles + obj2.num_triangles);

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
