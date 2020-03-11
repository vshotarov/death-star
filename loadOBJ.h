#ifndef LOAD_OBJ_H
#define LOAD_OBJ_H

#include "material.h"
#include "hittable.h"

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include <tiny_obj_loader.h>

__global__
void create_obj_hittables(Hittable* hittables, int num_hittables,
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

	int threads = 512;
	int dims = (scene.num_hittables + threads - 1) / threads;

	Material* material;
	
	cudaMalloc(&(material), 1 * sizeof(Material));

	initialize_obj_material<<<1, 1>>>(material);

	create_obj_hittables<<<dims, threads>>>(scene.hittables, scene.num_hittables,
			material, num_shapes, d_shape_sizes, d_indices, vertices, normals);
	cudaDeviceSynchronize();

    cudaError cudaErr = cudaGetLastError();
    if ( cudaSuccess != cudaErr )
    {
        fprintf( stderr, "cudaCheckError() failed at : %s\n",
                 cudaGetErrorString( cudaErr ) );
        exit( -1 );
    }
}

#endif
