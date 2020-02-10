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

enum class bvh_node_type
{
	internal,
	leaf
};

struct BVHNode
{
	__device__ static BVHNode internal()
	{
		BVHNode node;
		node.type = bvh_node_type::internal;
		node.atomic_visited_flag = 0;
		return node;
	}

	__device__ static BVHNode leaf()
	{
		BVHNode node;
		node.type = bvh_node_type::leaf;
		node.atomic_visited_flag = 0;
		return node;
	}

	bvh_node_type type;
	int atomic_visited_flag;
	BVHNode *parent;
	BVHNode* left;
	BVHNode* right;
	Hittable* hittable;

	AABB bounding_box;
};

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
void initialize_bvh_construction(Hittable** hittables, HittableWorld** world, int num_hittables,
		unsigned int *morton_codes, unsigned int *sorted_IDs, BVHNode *internal_nodes,
		BVHNode *leaf_nodes)
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

	leaf_nodes[idx] = BVHNode::leaf();
	if(idx < (num_hittables - 1))
		internal_nodes[idx] = BVHNode::internal();
}

__global__
void build_BVH_tree(Hittable** hittables, int num_hittables,
		unsigned int *morton_codes, unsigned int *sorted_IDs,
		BVHNode* internal_nodes, BVHNode* leaf_nodes)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < num_hittables)
		leaf_nodes[idx].hittable = hittables[sorted_IDs[idx]];

	if(idx >= num_hittables - 1)
		return;

	uint2 range = determine_range(morton_codes, idx, num_hittables);
	int split_position = find_split(morton_codes, range, idx, num_hittables);

	if(split_position == range.x)
		internal_nodes[idx].left = &leaf_nodes[split_position];
	else
		internal_nodes[idx].left = &internal_nodes[split_position];
	if(split_position + 1 == range.y)
		internal_nodes[idx].right = &leaf_nodes[split_position + 1];
	else
		internal_nodes[idx].right = &internal_nodes[split_position + 1];

	internal_nodes[idx].left->parent = &internal_nodes[idx];
	internal_nodes[idx].right->parent = &internal_nodes[idx];

	if(idx == 0)
		internal_nodes[idx].parent = NULL;
}

__global__
void print_bvh(BVHNode *internal_nodes, BVHNode *leaf_nodes, Hittable** hittables)
{
	for(int i=0; i<8; i++)
	{
		printf("Internal %i\n", i);
		printf("	bbox\n");
		printf("		min(%.2f, %.2f, %.2f)\n",
				internal_nodes[i].bounding_box.min.x,
				internal_nodes[i].bounding_box.min.y,
				internal_nodes[i].bounding_box.min.z);
		printf("		max(%.2f, %.2f, %.2f)\n",
				internal_nodes[i].bounding_box.max.x,
				internal_nodes[i].bounding_box.max.y,
				internal_nodes[i].bounding_box.max.z);
		if(internal_nodes[i].left->type == bvh_node_type::leaf)
		{
			printf("	left leaf\n");
			printf("		Sphere bbox->centroid (%.2f, %.2f, %.2f)\n",
					internal_nodes[i].left->hittable->bounding_box->centroid.x,
					internal_nodes[i].left->hittable->bounding_box->centroid.y,
					internal_nodes[i].left->hittable->bounding_box->centroid.z);
		}
		else
		{
			printf("	left internal\n");
		}
		if(internal_nodes[i].right->type == bvh_node_type::leaf)
		{
			printf("	right leaf\n");
			printf("		Sphere bbox->centroid (%.2f, %.2f, %.2f)\n",
					internal_nodes[i].right->hittable->bounding_box->centroid.x,
					internal_nodes[i].right->hittable->bounding_box->centroid.y,
					internal_nodes[i].right->hittable->bounding_box->centroid.z);
		}
		else
		{
			printf("	right internal\n");
		}
	}
}

__global__
void calculate_BVH_bounding_boxes(BVHNode* leaf_nodes, Hittable** hittables, int num_hittables)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= num_hittables)
		return;

	// Assign the bounding box of the hittable to the leaf
	leaf_nodes[idx].bounding_box = AABB(
			leaf_nodes[idx].hittable->bounding_box->min,
			leaf_nodes[idx].hittable->bounding_box->max);

	BVHNode* this_node = leaf_nodes[idx].parent;
	while(this_node != NULL)
	{
		if(atomicCAS(&(this_node->atomic_visited_flag), 0, 1))
		{
			// If this is the second time we are visiting this node,
			// that means all of it's children have been computed and
			// we can safely compute this one's bounding box
			this_node->bounding_box = surrounding_box(
					this_node->left->bounding_box,
					this_node->right->bounding_box);

			this_node = this_node->parent;
		}
		else
		{
			// This is the first time we are visiting this node, which
			// means that some of it's children have not been computed yes,
			// so we kill this thread, but next time this node is visited it will
			// be computed
			return;
		}
	}
}

BVHNode* create_BVH(Hittable** hittables, HittableWorld** world, int num_hittables)
{
	int dims = 8;
	int threads = 8;

	// Initialize memory for bvh nodes
	BVHNode *internal_nodes, *leaf_nodes;
	cudaMalloc(&internal_nodes, (num_hittables - 1) * sizeof(BVHNode));
	cudaMalloc(&leaf_nodes, num_hittables * sizeof(BVHNode));

	// Create morton codes for each centroid
	unsigned int *morton_codes, *sorted_IDs;
	cudaMalloc(&morton_codes, num_hittables * sizeof(unsigned int));
	cudaMalloc(&sorted_IDs, num_hittables * sizeof(unsigned int));

	initialize_bvh_construction<<<dims, threads>>>(hittables, world, num_hittables,
			morton_codes, sorted_IDs, internal_nodes, leaf_nodes);
	cudaDeviceSynchronize();

	// Sort morton codes
	thrust::sort_by_key(thrust::device,
			morton_codes, morton_codes + num_hittables, sorted_IDs);

	// Build the tree hierarchy
	build_BVH_tree<<<dims, threads>>>(hittables, num_hittables, morton_codes,
			sorted_IDs, internal_nodes, leaf_nodes);
	cudaDeviceSynchronize();

	calculate_BVH_bounding_boxes<<<dims, threads>>>(leaf_nodes, hittables, num_hittables);

	//print_bvh<<<1, 1>>>(internal_nodes, leaf_nodes, hittables);

	// NOTE: Figure out where to free internal_nodes and leaf_nodes cuda memory!!
	cudaFree(morton_codes);
	cudaFree(sorted_IDs);
	
	return &internal_nodes[0];
}

__device__ bool hit_BVH(BVHNode* root, const ray& r,
		float t_min, float t_max, hit_record& rec)
{
	// Allocate traversal stack from thread-local memory
	BVHNode* stack[64];
	BVHNode** stack_ptr = stack;
	*stack_ptr++ = NULL;

	BVHNode* node = root;
	bool any_hit = false;

	while(node != NULL)
	{
		bool hit_left = node->left->bounding_box.hit(r, t_min, t_max);
		bool hit_right = node->right->bounding_box.hit(r, t_min, t_max);

		bool local_hit = false;

		if(hit_left)
		{
			if(node->left->type == bvh_node_type::leaf)
				local_hit = node->left->hittable->hit(r, t_min, t_max, rec);
			else
				*stack_ptr++ = node->left;
		}

		if(hit_right)
		{
			if(node->right->type == bvh_node_type::leaf)
				local_hit = node->right->hittable->hit(r, t_min, t_max, rec);
			else
				*stack_ptr++ = node->right;
		}

		if(local_hit)
			any_hit = true;

		node = *--stack_ptr;
	}

	return any_hit;
}

#endif
