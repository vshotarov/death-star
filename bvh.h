/*
 * This is an implementation of the following paper by Tero karras
 * https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf
 *
 * Some of it is copied directly from the additional explanation
 * blog post https://devblogs.nvidia.com/thinking-parallel-part-iii-tree-construction-gpu/
*/
#ifndef BVH_H
#define BVH_H

#include "hittable.h"

#include <cstdio>

#include <thrust/sort.h>

#include <glm/glm.hpp>

using namespace glm;

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__ unsigned int expandBits(unsigned int v)
{
	// Copied directly from https://devblogs.nvidia.com/thinking-parallel-part-iii-tree-construction-gpu/
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ unsigned int morton3D(vec3 point)
{
	// Copied directly from https://devblogs.nvidia.com/thinking-parallel-part-iii-tree-construction-gpu/
    float x = min(max(point.x * 1024.0f, 0.0f), 1023.0f);
    float y = min(max(point.y * 1024.0f, 0.0f), 1023.0f);
    float z = min(max(point.z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

__device__ uint2 determine_range(unsigned int *morton_codes, int i, int num_leaf_nodes)
{
	if(i == 0)
	{
		return make_uint2(0, num_leaf_nodes-1);
	}

	int delta_left = -1;
	if(i > 0)
		delta_left = __clz(morton_codes[i] ^ morton_codes[i-1]);
	int delta_right = __clz(morton_codes[i] ^ morton_codes[i+1]);

	int d = delta_right > delta_left ? 1 : -1;

	// Compute upper bound for the length of the range
	int delta_min = delta_right > delta_left ? delta_left : delta_right;
	int lmax = 2;
	int delta = -1;
	int i_next = i + lmax * d;
	
	if(i_next >= 0 && i_next < num_leaf_nodes)
		delta = __clz(morton_codes[i] ^
					  morton_codes[i_next]);

	while(delta > delta_min)
	{
		lmax <<= 1; // Equivalent to lmax *= 2
		i_next = i + lmax * d;
		delta = -1;

		if(i_next >= 0 && i_next < num_leaf_nodes)
			delta = __clz(morton_codes[i] ^
						  morton_codes[i_next]);
	}

	// Find the other end using binary search
	int l = 0;
	int t = lmax >> 1; // Equivalent to t = lmax / 2;
	while(t > 0)
	{
		i_next = i + (l + t) * d;
		delta = -1;

		if(i_next >= 0 && i_next < num_leaf_nodes)
			delta = __clz(morton_codes[i] ^
						  morton_codes[i_next]);

		if(delta > delta_min)
		{
			l += t;
		}

		t >>= 1; // Equivalent to t /= 2;
	}

	unsigned int j = i + l * d;

	if(d < 0)
	{
		unsigned int tmp = j;
		j = i;
		i = tmp;
	}

	return make_uint2(i, j);
}

__device__ unsigned int find_split(unsigned int *morton_codes, const uint2& range,
		int i, int num_leaf_nodes)
{
	unsigned int first_code = morton_codes[range.x];
	unsigned int last_code = morton_codes[range.y];

	unsigned int split_position;

	if(first_code == last_code)
		return (range.x + range.y) >> 1; // Split in the middle

	int delta_node = __clz(first_code ^ last_code);

	split_position = range.x;
	int step = range.y - range.x;

	do
	{
		step = (step + 1) >> 1;
		int proposed_split = split_position + step;

		if(proposed_split < range.y)
		{
			unsigned int split_code = morton_codes[proposed_split];
			int split_prefix = __clz(first_code ^ split_code);
			if(split_prefix > delta_node)
				split_position = proposed_split;
		}
	} while(step > 1);

	return split_position;
}

__global__
void create_morton_codes(Hittable** hittables, HittableWorld** world, int num_hittables,
		unsigned int *morton_codes, unsigned int *sorted_IDs)
	// We pass sorted_IDs, as well, as it's a convinient way of constructing it
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= num_hittables)
		return;

	vec3 scene_size = (*world)->bounding_box.max
					- (*world)->bounding_box.min;

	morton_codes[idx] = morton3D((hittables[idx]->bounding_box->centroid
								- (*world)->bounding_box.min) / scene_size);
	sorted_IDs[idx] = idx;
}

__global__
void build_BVH_tree(Hittable** hittables, int num_hittables,
		unsigned int *morton_codes, unsigned int *sorted_IDs)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= num_hittables - 1)
		return;

	uint2 range = determine_range(morton_codes, idx, num_hittables);
	int split_position = find_split(morton_codes, range, idx, num_hittables);
}

void create_BVH(Hittable** hittables, HittableWorld** world, int num_hittables)
{
	// Create morton codes for each centroid
	unsigned int *morton_codes, *sorted_IDs;
	cudaMalloc(&morton_codes, num_hittables * sizeof(unsigned int));
	cudaMalloc(&sorted_IDs, num_hittables * sizeof(unsigned int));

	create_morton_codes<<<3, 3>>>(hittables, world, num_hittables, morton_codes, sorted_IDs);
	cudaDeviceSynchronize();

	// Sort morton codes
	thrust::sort_by_key(thrust::device,
			morton_codes, morton_codes + num_hittables, sorted_IDs);

	// Build the tree hierarchy
	build_BVH_tree<<<3, 3>>>(hittables, num_hittables, morton_codes, sorted_IDs);
}

#endif
