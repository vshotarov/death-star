#ifndef HIT_TESTS_H
#define HIT_TESTS_H


#include <glm/glm.hpp>

using namespace glm;

// Hit record used for storing and sharing intersection data
struct Material;
struct hit_record
{
	float t;
	vec3 p;
	vec3 normal;
	Material* material;
};

__device__
bool hit_sphere(const vec3& center, float radius, const ray& r, float t_min,
		float t_max, hit_record& rec)
{
	vec3 oc = r.origin - center;
	float a = dot(r.direction, r.direction);
	float b = dot(oc, r.direction);
	float c = dot(oc, oc) - radius*radius;
	float discriminant = b*b - a*c;

	if(discriminant > 0)
	{
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min)
		{
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min)
		{
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			return true;
		}
	}
	return false;
}

__device__
bool hit_triangle(const vec3& A, const vec3& B, const vec3& C, const vec3& normal,
		const ray& r, float t_min, float t_max, hit_record& rec)
{
	// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
	const float EPSILON = .0000001;
	vec3 edge1, edge2, h, s, q;
	float a, f, u, v;

	edge1 = B - A;
	edge2 = C - A;
	h = cross(r.direction, edge2);
	a = dot(edge1, h);

	if (a > -EPSILON && a < EPSILON)
		return false;  // Parallel to the triangle

	f = 1.0 / a;
	s = r.origin - A;
	u = f * dot(s, h);

	if (u < 0.0 || u > 1.0)
		return false;

	q = cross(s, edge1);
	v = f * dot(r.direction, q);

	if (v < 0.0 || u + v > 1.0)
		return false;

	float t = f * dot(edge2, q);

	if (t > t_min && t < t_max)
	{
		rec.t = t;
		rec.p = r.point_at_parameter(t);
		rec.normal = normal;
		return true;
	}
	else
		return false;
}

__device__ bool hit_AABB(const vec3& min, const vec3& max, const ray& r,
		float t_min, float t_max)
{
	for(int a=0; a<3; a++)
	{
		float invD = 1.0f / r.direction[a];
		float t0 = (min[a] - r.origin[a]) * invD;
		float t1 = (max[a] - r.origin[a]) * invD;

		if (invD < 0.0f)
		{
			float tmp = t0;
			t0 = t1;
			t1 = tmp;
		}

		t_min = t0 > t_min ? t0 : t_min;
		t_max = t1 < t_max ? t1 : t_max;

		//if(t_max <= t_min)
		// NOTE: I've adjusted this to allow fully flat bounding
		// boxes (bounding planes really)
		if(t_max < t_min)
			return false;
	}

	return true;
}

#endif
