#ifndef LOAD_OBJ_H
#define LOAD_OBJ_H

#include "material.h"
#include "hittable.h"

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include <tiny_obj_loader.h>


struct shapeData
{
	int size;
	tinyobj::index_t* indices;
};

struct objData
{
	shapeData* shapes;
	float* vertices;
	float* normals;
	int num_triangles;
	int num_shapes;
};

__global__
void create_obj_hittables(Hittable* hittables, Material* material,
		objData obj, int start_id)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= obj.num_triangles)
		return;

	// Identify triangle ID
	int tri_count = 0;
	int tri_id = 0;
	int shape_id = 0;
	for(int s=0; s<obj.num_shapes; s++)
	{
		if(idx < tri_count + obj.shapes[s].size)
		{
			tri_id = idx - tri_count;
			shape_id = s;
			break;
		}
		tri_count += obj.shapes[s].size;
	}

	// Triangles
	float triangle_points[9];
	for(int v=0; v<3; v++)
	{
		tinyobj::index_t idx = obj.shapes[shape_id].indices[tri_id*3 + v];
		triangle_points[v*3 + 0] = obj.vertices[3*idx.vertex_index+0];
		triangle_points[v*3 + 1] = obj.vertices[3*idx.vertex_index+1];
		triangle_points[v*3 + 2] = obj.vertices[3*idx.vertex_index+2];
	}

	hittables[start_id + idx] = Hittable::triangle(
			vec3(triangle_points[0], triangle_points[1], triangle_points[2]),
			vec3(triangle_points[3], triangle_points[4], triangle_points[5]),
			vec3(triangle_points[6], triangle_points[7], triangle_points[8]),
			material);
}

__global__
void initialize_obj_material(Material* material)
{
	*material = *Material::lambertian(vec3(.5, .3, .1));
}

objData load_obj(const char* obj_file)
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

	objData data;
	data.num_triangles = 0;
	data.num_shapes = shapes.size();

	std::vector<shapeData> d_shapes;
	
	for(int i=0; i<data.num_shapes; i++)
	{
		// We only accept triangles, but we don't check for them,
		// we just assume that's what we get
		shapeData shape;
		shape.size = shapes[i].mesh.num_face_vertices.size();
		data.num_triangles += shape.size;

		cudaMalloc((tinyobj::index_t**)&shape.indices,
				   shapes[i].mesh.indices.size() * sizeof(tinyobj::index_t));
		cudaMemcpy(shape.indices, &(shapes[i].mesh.indices[0]),
				   shapes[i].mesh.indices.size() * sizeof(tinyobj::index_t),
				   cudaMemcpyHostToDevice);

		d_shapes.push_back(shape);
	}

	printf("num shapes %i, num hittables %i, num shapes %i\n",
			data.num_shapes, data.num_triangles, data.num_shapes);

	cudaMalloc((shapeData**)&data.shapes, data.num_shapes * sizeof(shapeData));
	cudaMalloc((float**)&data.vertices, attrib.vertices.size() * sizeof(float));
	cudaMalloc((float**)&data.normals, attrib.normals.size() * sizeof(float));

	cudaMemcpy(data.shapes, &(d_shapes[0]), data.num_shapes * sizeof(shapeData), cudaMemcpyHostToDevice);
	cudaMemcpy(data.vertices, &(attrib.vertices[0]), attrib.vertices.size() * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(data.normals, &(attrib.normals[0]), attrib.normals.size() * sizeof(float), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
    cudaError cudaErr = cudaGetLastError();
    if ( cudaSuccess != cudaErr )
    {
        fprintf( stderr, "cudaCheckError() failed at copying obj to device: %s\n",
                 cudaGetErrorString( cudaErr ) );
        exit( -1 );
    }

	return data;
}

#endif
