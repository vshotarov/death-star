#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"

#include <glm/glm.hpp>

using namespace glm;

// Hit record used for storing and sharing intersection data
struct Material;  // Define Material, so it can be stored on Hittables

struct hit_record
{
	float t;
	vec3 p;
	vec3 normal;
	Material* material;
};

// From here on we define all hittable types, and in the end
// the base Hittable struct, which will be able to represent
// a hittable of certain type
enum class hittable_type
{
	// hittable_type used in union to specify the type of
	// hittable a Hittable struct is representing
	sphere,
	triangle
};

struct Sphere
{
	public:
		__device__ Sphere() {};
		__device__ Sphere(vec3 center, float radius) :
			center(center), radius(radius) {}
		__device__ void destroy() {};

		__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec)
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

	private:
		vec3 center;
		float radius;
};

struct Triangle
{
	public:
		__device__ Triangle() {};
		__device__ Triangle(vec3 A, vec3 B, vec3 C) :
			A(A), B(B), C(C)
		{
			normal = normalize(cross(B-A, C-A));
		};

		__device__ void destroy() {};

		__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec)
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

	private:
		vec3 A;
		vec3 B;
		vec3 C;
		vec3 normal;
};

struct Hittable
{
	// Thanks to the discriminating union pattern we can represent
	// different hittable types without polymorphism
	public:
		__device__ Hittable(hittable_type type, Material* material) :
			_type(type), material(material) {}
		__device__ hittable_type type() { return _type; }
		__device__ ~Hittable() { destroy(); }

		__device__ static Hittable* sphere(vec3 center, float radius,
				Material* material)
		{
			Hittable* hittable = new Hittable(hittable_type::sphere,
											  material);
			hittable->_sphere = Sphere(center, radius);
			return hittable;
		}

		__device__ static Hittable* triangle(vec3 a, vec3 b, vec3 c,
				Material* material)
		{
			Hittable* hittable = new Hittable(hittable_type::triangle,
											  material);
			hittable->_triangle = Triangle(a, b, c);
			return hittable;
		}

		__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec)
		{
			bool return_val;

			switch(_type)
			{
				case hittable_type::sphere:
					return_val = _sphere.hit(r, t_min, t_max, rec); break;
				case hittable_type::triangle:
					return_val = _triangle.hit(r, t_min, t_max, rec); break;
			}

			if(return_val)
				rec.material = material;

			return return_val;
		}

	private:
		hittable_type _type;
		Material* material;

		union {
			Sphere _sphere;
			Triangle _triangle;
		};

		__device__ void destroy()
		{
			switch(_type)
			{
				case hittable_type::sphere: _sphere.destroy(); break;
				case hittable_type::triangle: _triangle.destroy(); break;
			}
		}
};

struct HittableWorld
{
	public:
		__device__ HittableWorld(Hittable** hittables, int num_hittables) :
			hittables(hittables), num_hittables(num_hittables) {};
		__device__ ~HittableWorld() { destroy(); }

		__device__ void destroy()
		{
			delete [] hittables;
		};

		__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec)
		{
			bool any_hit = false;
			float this_t_max = t_max;

			for(int i=0; i<num_hittables; i++)
			{
				if(hittables[i]->hit(r, t_min, this_t_max, rec))
				{
					this_t_max = rec.t;
					any_hit = true;
				}
			}

			return any_hit;
		}

		__device__ int size() { return num_hittables; }

	private:
		int num_hittables;
		Hittable** hittables;
};

#endif
