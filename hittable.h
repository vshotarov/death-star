#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"
#include "AABB.h"
#include "hit_tests.h"

#include <glm/glm.hpp>

using namespace glm;

struct Material;  // Define Material, so it can be stored on Hittables

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
	vec3 center;
	float radius;
};

struct Triangle
{
	vec3 A;
	vec3 B;
	vec3 C;
	vec3 normal;
};

class Hittable
{
	// Thanks to the discriminating union pattern we can represent
	// different hittable types without polymorphism
	public:
		__device__ Hittable(hittable_type type, Material* material) :
			_type(type), material(material) {}
		__device__ ~Hittable() {}
		__device__ hittable_type type() {return _type;}
		__device__ static Hittable sphere(vec3 center, float radius,
				Material* material = NULL)
		{
			Hittable hittable = Hittable(hittable_type::sphere, material);
			hittable._sphere = Sphere();
			hittable._sphere.center = center;
			hittable._sphere.radius = radius;
			hittable.bounding_box = AABB(vec3(center.x-radius, center.y-radius, center.z-radius),
										 vec3(center.x+radius, center.y+radius, center.z+radius));
			return hittable;
		}

		__device__ static Hittable triangle(vec3 A, vec3 B, vec3 C,
				Material* material = NULL)
		{
			Hittable hittable = Hittable(hittable_type::triangle, material);
			hittable._triangle = Triangle();
			hittable._triangle.A = A;
			hittable._triangle.B = B;
			hittable._triangle.C = C;
			hittable._triangle.normal = normalize(cross(B-A, C-A));

			float sx = A.x < B.x ? A.x : B.x;
			sx = C.x < sx ? C.x : sx;
			float sy = A.y < B.y ? A.y : B.y;
			sy = C.y < sy ? C.y : sy;
			float sz = A.z < B.z ? A.z : B.z;
			sz = C.z < sz ? C.z : sz;

			float lx = A.x > B.x ? A.x : B.x;
			lx = C.x > lx ? C.x : lx;
			float ly = A.y > B.y ? A.y : B.y;
			ly = C.y > ly ? C.y : ly;
			float lz = A.z > B.z ? A.z : B.z;
			lz = C.z > lz ? C.z : lz;

			hittable.bounding_box = AABB(vec3(sx,sy,sz), vec3(lx,ly,lz));

			return hittable;
		}

		__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec)
		{
			bool return_val;

			switch(_type)
			{
				case hittable_type::sphere:
					return_val = hit_sphere(_sphere.center, _sphere.radius, r,
							t_min, t_max, rec); break;
				case hittable_type::triangle:
					return_val = hit_triangle(_triangle.A, _triangle.B,
							_triangle.C, _triangle.normal, r, t_min, t_max,
							rec); break;
			}

			if(return_val)
				rec.material = material;

			return return_val;
		}

		AABB bounding_box;

	private:
		hittable_type _type;
		Material* material;

		union
		{
			Sphere _sphere;
			Triangle _triangle;
		};
};
#endif
