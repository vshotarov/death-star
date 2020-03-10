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

__device__ int common_prefix(unsigned int* morton_codes, int i, int j, int num_hittables)
{
	if(i < 0 || i >= num_hittables || j < 0 || j >= num_hittables)
		return -1;

	unsigned int morton_i = morton_codes[i];
	unsigned int morton_j = morton_codes[j];

	if(morton_i == morton_j)
		return __clzll(morton_i ^ morton_j) + __clzll(i ^ j);
	else
		return __clzll(morton_i ^ morton_j);
}

enum class bvh_node_type
{
	internal,
	leaf
};

struct __align__(32) BVHNode
{
	__device__ static BVHNode internal()
	{
		BVHNode node;
		node.type = bvh_node_type::internal;
		node.atomic_visited_flag = 0;
		node.parent = NULL;
		node.hittable = NULL;
		return node;
	}

	__device__ static BVHNode leaf()
	{
		BVHNode node;
		node.type = bvh_node_type::leaf;
		node.atomic_visited_flag = 0;
		node.parent = NULL;
		node.hittable = NULL;
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

	int d = (common_prefix(morton_codes, i, i+1, num_leaf_nodes)
		   - common_prefix(morton_codes, i, i-1, num_leaf_nodes)) >= 0 ? 1 : -1;

	// Compute upper bound for the length of the range
	int delta_min = common_prefix(morton_codes, i, i-d, num_leaf_nodes);
	int lmax = 2;

	while(common_prefix(morton_codes, i, i + lmax*d, num_leaf_nodes) > delta_min)
		lmax *= 2;

	// Find the other end using binary search
	int l = 0;
	int t = lmax >> 1; // Equivalent to t = lmax / 2;

	while(t > 0)
	{
		if(common_prefix(morton_codes, i, i + (l + t) * d, num_leaf_nodes) > delta_min)
			l += t;
		t /= 2;
	}

	unsigned int j = i + l * d;

	return make_uint2(i, j);
}

__device__ unsigned int find_split(unsigned int *morton_codes, const uint2& range,
		int i, int num_leaf_nodes)
{
	// Originally I wrote this using the paper as a reference, but I couldn't get
	// it to work properly when there were multiple hittable duplicates.
	// Borrowing the code from https://github.com/henrikdahlberg/GPUPathTracer/blob/master/Source/Core/BVHConstruction.cu
	// made it work, even though that loop is quite odd as it runs twice with
	// t = (l + (divider - 1)) / divider  , where divider is 2

	int delta_node = common_prefix(morton_codes, range.x, range.y, num_leaf_nodes);
	int d = range.y >= range.x ? 1 : -1;
	int s = 0;
	int l = range.y >= range.x ? range.y - range.x : range.x - range.y;
	int divider = 2;

	for(int t=(l+divider-1)/divider; t>0; divider*=2)
	{
		if(common_prefix(morton_codes, range.x, range.x + (s + t) * d,
					num_leaf_nodes) > delta_node)
			s += t;
		if(t==1) break;
		t = (l + divider - 1) / divider;
	}

	return range.x + s * d + min(d, 0);
}

__global__
void calculate_world_bounding_box(Hittable* hittables, int num_hittables,
		AABB* world_bounding_box)
{
	world_bounding_box->min = hittables[0].bounding_box.min;
	world_bounding_box->max = hittables[0].bounding_box.max;
	if(num_hittables > 1)
		for(int i=1; i<num_hittables; i++)
			*world_bounding_box = surrounding_box(*world_bounding_box,
											      hittables[i].bounding_box);
}

__global__
void initialize_bvh_construction(Hittable* hittables, int num_hittables, AABB* world_bounding_box,
		unsigned int *morton_codes, unsigned int *sorted_IDs, BVHNode *internal_nodes,
		BVHNode *leaf_nodes)
	// We pass sorted_IDs, as well, as it's a convinient way of constructing it
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= num_hittables)
		return;

	// Calculate all hittables merged bounding box
	vec3 scene_size = world_bounding_box->max
					- world_bounding_box->min;

	morton_codes[idx] = morton3D((hittables[idx].bounding_box.centroid
								- world_bounding_box->min) / scene_size);
	sorted_IDs[idx] = idx;

	leaf_nodes[idx] = BVHNode::leaf();
	if(idx < (num_hittables - 1))
		internal_nodes[idx] = BVHNode::internal();
}

__global__
void build_BVH_tree(Hittable* hittables, int num_hittables,
		unsigned int *morton_codes, unsigned int *sorted_IDs,
		BVHNode* internal_nodes, BVHNode* leaf_nodes)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < num_hittables)
		leaf_nodes[idx].hittable = &(hittables[sorted_IDs[idx]]);

	if(idx >= num_hittables - 1)
		return;

	uint2 range = determine_range(morton_codes, idx, num_hittables);
	int split_position = find_split(morton_codes, range, idx, num_hittables);

	if(split_position == min(range.x, range.y))
		internal_nodes[idx].left = &leaf_nodes[split_position];
	else
		internal_nodes[idx].left = &internal_nodes[split_position];
	if(split_position + 1 == max(range.x, range.y))
		internal_nodes[idx].right = &leaf_nodes[split_position + 1];
	else
		internal_nodes[idx].right = &internal_nodes[split_position + 1];

	internal_nodes[idx].left->parent = &internal_nodes[idx];
	internal_nodes[idx].right->parent = &internal_nodes[idx];

	if(idx == 0)
		internal_nodes[idx].parent = NULL;
}

__global__
void print_bvh(BVHNode *internal_nodes, BVHNode *leaf_nodes, Hittable* hittables)
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
					internal_nodes[i].left->hittable->bounding_box.centroid.x,
					internal_nodes[i].left->hittable->bounding_box.centroid.y,
					internal_nodes[i].left->hittable->bounding_box.centroid.z);
		}
		else
		{
			printf("	left internal\n");
		}
		if(internal_nodes[i].right->type == bvh_node_type::leaf)
		{
			printf("	right leaf\n");
			printf("		Sphere bbox->centroid (%.2f, %.2f, %.2f)\n",
					internal_nodes[i].right->hittable->bounding_box.centroid.x,
					internal_nodes[i].right->hittable->bounding_box.centroid.y,
					internal_nodes[i].right->hittable->bounding_box.centroid.z);
		}
		else
		{
			printf("	right internal\n");
		}
	}
}

__global__
void calculate_BVH_bounding_boxes(BVHNode* leaf_nodes, Hittable* hittables, int num_hittables)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= num_hittables)
		return;

	// Assign the bounding box of the hittable to the leaf
	leaf_nodes[idx].bounding_box = AABB(
			leaf_nodes[idx].hittable->bounding_box.min,
			leaf_nodes[idx].hittable->bounding_box.max);

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

BVHNode* create_BVH(Hittable* hittables, int num_hittables)
{
	int threads = 512;
	int dims = (num_hittables + threads - 1) / threads;

	// Initialize memory for bvh nodes
	BVHNode *internal_nodes, *leaf_nodes;
	cudaMalloc(&internal_nodes, (num_hittables - 1) * sizeof(BVHNode));
	cudaMalloc(&leaf_nodes, num_hittables * sizeof(BVHNode));

	// Calcluate world bounding box, as it's needed for localizing morton codes
	AABB* world_bounding_box;
	cudaMalloc(&world_bounding_box, sizeof(AABB));
	calculate_world_bounding_box<<<1, 1>>>(hittables, num_hittables, world_bounding_box);

	// Create morton codes for each centroid
	unsigned int *morton_codes, *sorted_IDs;
	cudaMalloc(&morton_codes, num_hittables * sizeof(unsigned int));
	cudaMalloc(&sorted_IDs, num_hittables * sizeof(unsigned int));

	initialize_bvh_construction<<<dims, threads>>>(hittables, num_hittables,
			world_bounding_box, morton_codes, sorted_IDs, internal_nodes, leaf_nodes);
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
	float this_t_max = t_max;

	while(node != NULL)
	{
		bool hit_left = node->left->bounding_box.hit(r, t_min, this_t_max);
		bool hit_right = node->right->bounding_box.hit(r, t_min, this_t_max);

		bool local_hit = false;

		if(hit_left)
		{
			if(node->left->type == bvh_node_type::leaf)
			{
				local_hit = node->left->hittable->hit(r, t_min, this_t_max, rec);
				if(local_hit)
				{
					this_t_max = rec.t;
					any_hit = true;
				}
			}
			else
				*stack_ptr++ = node->left;
		}

		if(hit_right)
		{
			if(node->right->type == bvh_node_type::leaf)
			{
				local_hit = node->right->hittable->hit(r, t_min, this_t_max, rec);
				if(local_hit)
				{
					this_t_max = rec.t;
					any_hit = true;
				}
			}
			else
				*stack_ptr++ = node->right;
		}

		node = *--stack_ptr;
	}

	return any_hit;
}

#endif
