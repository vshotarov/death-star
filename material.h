#ifndef MATERIAL_H
#define MATERIAL_H

#include "ray.h"
#include "random.h"
#include "hittable.h"

#include <glm/glm.hpp>

using namespace glm;


// Material types
enum class material_type
{
	lambertian
};

struct Lambertian
{
	public:
		__device__ Lambertian(vec3 albedo) : albedo(albedo) {};
		__device__ void destroy() {};

		__device__ bool scatter(const ray& r, const hit_record& rec, vec3& attenuation,
				ray& scattered, curandState* rand_state)
		{
			vec3 bounced_ray_dir = rec.normal + random_in_unit_sphere(rand_state);

			scattered = ray(rec.p, bounced_ray_dir);

			attenuation = albedo;

			return true;
		}

	private:
		vec3 albedo;
};

struct Material
{
	public:
		__device__ Material(material_type type) : _type(type) {}
		__device__ material_type type() { return _type; }
		__device__ ~Material() { destroy(); }

		__device__ static Material* lambertian(vec3 albedo)
		{
			Material* material = new Material(material_type::lambertian);
			material->_lambertian = Lambertian(albedo);
			return material;
		}

		__device__ bool scatter(const ray& r, const hit_record& rec, vec3& attenuation,
				ray& scattered, curandState* rand_state)
		{
			bool out;

			switch(_type)
			{
				case material_type::lambertian: out = _lambertian.scatter(
					r, rec, attenuation, scattered, rand_state); break;
			}

			return out;
		}

		__device__ void destroy()
		{
			switch(_type)
			{
				case material_type::lambertian:
				{
					_lambertian.destroy();
					break;
				}
			}
		}

	private:
		material_type _type;

		union
		{
			Lambertian _lambertian;
		};
};

#endif
