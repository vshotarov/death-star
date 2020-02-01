#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"

#include <glm/glm.hpp>

using namespace glm;

// Hit record used for storing and sharing intersection data
struct hit_record
{
	float t;
	vec3 p;
	vec3 normal;
};

// From here on we define all hittable types, and in the end
// the base Hittable struct, which will be able to represent
// a hittable of certain type
struct Hittable;

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
		Sphere() {};
		Sphere(vec3 center, float radius) :
			center(center), radius(radius) {}
		void destroy() {};

		bool hit(const ray& r, float t_min, float t_max, hit_record& rec)
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
		Triangle() {};
		Triangle(vec3 A, vec3 B, vec3 C) :
			A(A), B(B), C(C)
		{
			normal = normalize(cross(A, B));
		};

		void destroy() {};

		bool hit(const ray& r, float t_min, float t_max, hit_record& rec)
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

			/* NOTE in the original algorithm they use the following if
			 * statement to differentiate line and ray intersections, which
			 * I don't need here, but if any errors pop up down the line
			 * it's worth investigating
			if (t > EPSILON && t < (1 - EPSILON))
			{
				rec.t = t;
				rec.p = r.point_at_parameter(t);
				rec.normal = normal;
				return true;
			}
			else
				return false;
			*/

			rec.t = t;
			rec.p = r.point_at_parameter(t);
			rec.normal = normal;
			return true;
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
		Hittable(hittable_type type) : _type(type) {}
		hittable_type type() { return _type; }
		~Hittable() { destroy(); }

		static Hittable sphere(vec3 center, float radius)
		{
			Hittable hittable(hittable_type::sphere);
			hittable._sphere = Sphere(center, radius);
			return hittable;
		}

		static Hittable triangle(vec3 a, vec3 b, vec3 c)
		{
			Hittable hittable(hittable_type::triangle);
			hittable._triangle = Triangle(a, b, c);
			return hittable;
		}

		bool hit(const ray& r, float t_min, float t_max, hit_record& rec)
		{
			bool return_val;

			switch(_type)
			{
				case hittable_type::sphere:
					return_val = _sphere.hit(r, t_min, t_max, rec); break;
				case hittable_type::triangle:
					return_val = _triangle.hit(r, t_min, t_max, rec); break;
			}

			return return_val;
		}

	private:
		hittable_type _type;

		union {
			Sphere _sphere;
			Triangle _triangle;
		};

		void destroy()
		{
			switch(_type)
			{
				case hittable_type::sphere: _sphere.destroy(); break;
				case hittable_type::triangle: _triangle.destroy(); break;
			}
		}
};

#endif
