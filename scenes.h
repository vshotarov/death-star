#ifndef SCENES_H
#define SCENES_H

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "tiny_obj_loader.h"
#include <iostream>

#include "material.h"

#include <curand_kernel.h>

struct Scene
{
	int num_hittables;
	Hittable* hittables;
	HittableWorld* world;
};

__global__
void create_RTOW_three_spheres_on_top_of_big_sphere_scene(Hittable* hittables,
		HittableWorld* world)
{
	hittables[0] = Hittable::sphere(vec3(.0, .0, -1.0), .5,
			Material::lambertian(vec3(.8, .3, .3)));
	hittables[1] = Hittable::sphere(vec3(.0, -100.5, -1.0), 100,
			Material::lambertian(vec3(.8, .8, .0)));
	hittables[2] = Hittable::sphere(vec3(1.0, 0.0, -1.0), .5,
			Material::metal(vec3(.8, .6, .2), 0.0f));
	hittables[3] = Hittable::sphere(vec3(-1.0, 0.0, -1.0), .5,
			Material::metal(vec3(.8, .8, .8), 0.3f));

	*world = HittableWorld(hittables, 4);
}

__global__
void create_sphere_on_top_of_big_sphere_scene(Hittable* hittables,
		HittableWorld* world)
{
	Material* material = Material::lambertian(vec3(.5, .5, .5));

	hittables[0] = Hittable::sphere(vec3(.0, .0, -1.0), .5, material);
	hittables[1] = Hittable::sphere(vec3(.0, -100.5, -1.0), 100, material);

	*world = HittableWorld(hittables, 2);
}

__global__
void create_sphere_and_two_triangles_scene(Hittable* hittables,
		HittableWorld* world)
{
	Material* material = Material::lambertian(vec3(.5, .5, .5));

	hittables[0] = Hittable::sphere(vec3(.0, .0, -1.0), .5, material);
	hittables[1] = Hittable::triangle(vec3(-1.5f, 0.0f, -1.0f),
								  vec3(-2.0f, 1.0f, -2.0f),
								  vec3(-3.0f, 0.0f, -3.0f), material);
	hittables[2] = Hittable::triangle(vec3(.0f, 0.0f, -1.0f),
								  vec3(-.5f, 0.5f, -1.0f),
								  vec3(-1.5f, -.2f, -1.0f), material);

	*world = HittableWorld(hittables, 3);
}

__global__
void create_random_spheres_and_triangles_scene(Hittable* hittables,
		HittableWorld* world, int num_hittables = 50)
{
	curandState rand_state;
	int seed = 2020;
	curand_init(seed, 0, 0, &rand_state);

	for(int i=0; i<num_hittables; i++)
	{
		// half spheres and half triangles
		if(curand_uniform(&rand_state) > .5f)
		{
			Material* material = Material::metal(vec3(.5, curand_uniform(&rand_state), .5),
					curand_uniform(&rand_state) * .2f);
			hittables[i] = Hittable::sphere(
					4.0f * (vec3(curand_uniform(&rand_state),
						         curand_uniform(&rand_state),
						         curand_uniform(&rand_state) - 1.0f) - .5f),
					curand_uniform(&rand_state), material);
		}
		else
		{
			Material* material = Material::lambertian(vec3(
						curand_uniform(&rand_state), curand_uniform(&rand_state), .5));
			hittables[i] = Hittable::triangle(
					4.0f * (vec3(curand_uniform(&rand_state),
						 	     curand_uniform(&rand_state),
						 	     curand_uniform(&rand_state) - 1.0f) - .5f),
					4.0f * (vec3(curand_uniform(&rand_state),
						 	     curand_uniform(&rand_state),
						 	     curand_uniform(&rand_state) - 1.0f) - .5f),
					4.0f * (vec3(curand_uniform(&rand_state),
						 	     curand_uniform(&rand_state),
						 	     curand_uniform(&rand_state) - 1.0f) - .5f),
					material);
		}
	}

	*world = HittableWorld(hittables, num_hittables);
}

__global__
void create_BVH_test_scene(Hittable* hittables, HittableWorld* world)
{
	Material* material = Material::lambertian(vec3(.5, .5, .5));
	for(int i=0; i<3; i++)
	{
		for(int j=0; j<3; j++)
		{
			hittables[i*3+j] = Hittable::sphere(vec3((i-1)*2,(j-1)*2,-1.0), .5, material);
		}
	}
	*world = HittableWorld(hittables, 9);
}

void create_custom_scene(Scene& scene)
{
	scene.num_hittables = 50;

	cudaMalloc(&(scene.hittables), scene.num_hittables * sizeof(Hittable));
	cudaMalloc(&(scene.world), 1 * sizeof(HittableWorld));

	create_random_spheres_and_triangles_scene<<<1, 1>>>(scene.hittables, scene.world);
}

__global__
void create_obj_hittables(Hittable* hittables, HittableWorld* world, int num_hittables,
	Material* material, int num_shapes, int* shape_sizes, tinyobj::index_t* indices,
	float* vertices, float* normals)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= num_hittables)
		return;

	// Identify triangle ID
	int tri_count = 0;
	int tri_id = 0;
	for(int s=0; s<num_shapes; s++)
	{
		if(idx < tri_count + shape_sizes[s])
		{
			tri_id = idx - tri_count;
			break;
		}
		tri_count += shape_sizes[s];
	}

	// Triangles
	float triangle_points[9];
	for(int v=0; v<3; v++)
	{
		tinyobj::index_t idx = indices[tri_id*3 + v];
		triangle_points[v*3 + 0] = vertices[3*idx.vertex_index+0];
		triangle_points[v*3 + 1] = vertices[3*idx.vertex_index+1];
		triangle_points[v*3 + 2] = vertices[3*idx.vertex_index+2];
	}

	hittables[idx] = Hittable::triangle(
			vec3(triangle_points[0], triangle_points[1], triangle_points[2]),
			vec3(triangle_points[3], triangle_points[4], triangle_points[5]),
			vec3(triangle_points[6], triangle_points[7], triangle_points[8]),
			material);

	//int hittable_idx = 0;
	//for(int s=0; s<num_shapes; s++)
	//{
	//	for(int f=0; f<shape_sizes[s]; f++)
	//	{
	//		float triangle_points[9];
	//		// Triangles
	//		for(int v=0; v<3; v++)
	//		{
	//			tinyobj::index_t idx = indices[f*3 + v];
	//			triangle_points[v*3 + 0] = vertices[3*idx.vertex_index+0];
	//			triangle_points[v*3 + 1] = vertices[3*idx.vertex_index+1];
	//			triangle_points[v*3 + 2] = vertices[3*idx.vertex_index+2];
	//		}

	//		hittables[hittable_idx] = Hittable::triangle(
	//				vec3(triangle_points[0], triangle_points[1], triangle_points[2]),
	//				vec3(triangle_points[3], triangle_points[4], triangle_points[5]),
	//				vec3(triangle_points[6], triangle_points[7], triangle_points[8]),
	//				material);

	//		hittable_idx ++;
	//	}

	//	printf("CUDA: shape %i, shape size %i\n", s, shape_sizes[s]);
	//}

	//*world = HittableWorld(hittables, hittable_idx);
}

__global__
void create_obj_world(Hittable* hittables, HittableWorld* world, int num_hittables)
{
	*world = HittableWorld(hittables, num_hittables);
}

__global__
void initialize_obj_material(Material* material)
{
	*material = *Material::lambertian(vec3(.5, .5, .5));
}

void load_obj(Scene& scene, const char* obj_file)
{
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string warn;
	std::string err;

	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, obj_file);

	if (!warn.empty()) {
	  std::cout << warn << std::endl;
	}

	if (!err.empty()) {
	  std::cerr << err << std::endl;
	}

	if (!ret) {
	  exit(1);
	}

	int num_shapes = shapes.size();
	std::vector<int> shape_sizes;
	std::vector<tinyobj::index_t> indices;
	scene.num_hittables = 0;
	for(int i=0; i<num_shapes; i++)
	{
		// We only accept triangles, but we don't check for them,
		// we just assume that's what we get
		shape_sizes.push_back(shapes[i].mesh.num_face_vertices.size());
		scene.num_hittables += shape_sizes[i];
		indices.insert(indices.end(), shapes[i].mesh.indices.begin(),
									  shapes[i].mesh.indices.end());
	}

	printf("num shapes %i, num hittables %i, num shape_sizes %i, num indices %i\n",
			num_shapes, scene.num_hittables, shape_sizes.size(), indices.size());

	int* d_shape_sizes;
	tinyobj::index_t* d_indices;
	float* vertices;
	float* normals;

	cudaMalloc((void**)&d_shape_sizes, shape_sizes.size() * sizeof(int));
	cudaMalloc((tinyobj::index_t**)&d_indices, indices.size() * sizeof(tinyobj::index_t));
	cudaMalloc((float**)&vertices, attrib.vertices.size() * sizeof(float));
	cudaMalloc((float**)&normals, attrib.normals.size() * sizeof(float));

	cudaMemcpy(d_shape_sizes, &(shape_sizes[0]), shape_sizes.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, &(indices[0]), indices.size() * sizeof(tinyobj::index_t), cudaMemcpyHostToDevice);
	cudaMemcpy(vertices, &(attrib.vertices[0]), attrib.vertices.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(normals, &(attrib.normals[0]), attrib.normals.size() * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&(scene.hittables), scene.num_hittables * sizeof(Hittable));
	cudaMalloc(&(scene.world), 1 * sizeof(HittableWorld));

	int threads = 512;
	int dims = (scene.num_hittables + threads - 1) / threads;

	Material* material;
	
	cudaMalloc(&(material), 1 * sizeof(Material));

	initialize_obj_material<<<1, 1>>>(material);

	create_obj_hittables<<<dims, threads>>>(scene.hittables, scene.world, scene.num_hittables,
			material, num_shapes, d_shape_sizes, d_indices, vertices, normals);
	cudaDeviceSynchronize();

	create_obj_world<<<1, 1>>>(scene.hittables, scene.world, scene.num_hittables);
	cudaDeviceSynchronize();

    cudaError cudaErr = cudaGetLastError();
    if ( cudaSuccess != cudaErr )
    {
        fprintf( stderr, "cudaCheckError() failed at : %s\n",
                 cudaGetErrorString( cudaErr ) );
        exit( -1 );
    }

	//exit(0);
}

#endif
