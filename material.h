#ifndef MATERIAL_H
#define MATERIAL_H

#include "ray.h"
#include "random.h"
#include "shading_utils.h"


// Material types
enum class material_type
{
	lambertian,
	metal,
	dielectric
};

struct Lambertian
{
	public:
		__device__ Lambertian(vec3 albedo) : albedo(albedo) {};
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

struct Metal
{
	public:
		__device__ Metal(vec3 albedo, float fuzz) :
			albedo(albedo), fuzz(fuzz) {};
		__device__ bool scatter(const ray& r, const hit_record& rec, vec3& attenuation,
				ray& scattered, curandState* rand_state)
		{
			vec3 reflected = reflect(r.direction, rec.normal);

			scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(rand_state));

			attenuation = albedo;

			return (dot(scattered.direction, rec.normal) > 0); // Kill ray
		}

	private:
		vec3 albedo;
		float fuzz;
};

struct Dielectric
{
	public:
		__device__ Dielectric(float refractive_idx) : refractive_idx(refractive_idx) {};
		__device__ bool scatter(const ray& r, const hit_record& rec, vec3& attenuation,
				ray& scattered, curandState* rand_state)
		{
            vec3 outward_normal;
            vec3 reflected = reflect(r.direction, rec.normal);
            float ni_over_nt;
            attenuation = vec3(1.0, 1.0, 1.0);
            vec3 refracted;

            float reflect_prob;
            float cosine;

            if (dot(r.direction, rec.normal) > 0) {
                 outward_normal = -rec.normal;
                 ni_over_nt = refractive_idx;
                 cosine = refractive_idx * dot(r.direction, rec.normal)
                        / r.direction.length();
            }
            else {
                 outward_normal = rec.normal;
                 ni_over_nt = 1.0 / refractive_idx;
                 cosine = -dot(r.direction, rec.normal)
                        / r.direction.length();
            }

            if (death_star::refract(r.direction, outward_normal, ni_over_nt, refracted)) {
               reflect_prob = schlick(cosine, refractive_idx);
            }
            else {
               reflect_prob = 1.0;
            }

            if (curand_uniform(rand_state) < reflect_prob) {
               scattered = ray(rec.p, reflected);
            }
            else {
               scattered = ray(rec.p, refracted);
            }

            return true;
        }
	private:
		float refractive_idx;
};

struct Material
{
	public:
		__device__ material_type type() { return _type; }

		__device__ static Material* lambertian(vec3 albedo)
		{
			Material* material = new Material(material_type::lambertian);
			material->_lambertian = Lambertian(albedo);
			return material;
		}

		__device__ static Material* metal(vec3 albedo, float fuzz)
		{
			Material* material = new Material(material_type::metal);
			material->_metal = Metal(albedo, fuzz);
			return material;
		}

		__device__ static Material* dielectric(float refractive_idx)
		{
			Material* material = new Material(material_type::dielectric);
			material->_dielectric = Dielectric(refractive_idx);
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
				case material_type::metal: out = _metal.scatter(
					r, rec, attenuation, scattered, rand_state); break;
				case material_type::dielectric: out = _dielectric.scatter(
					r, rec, attenuation, scattered, rand_state); break;
			}

			return out;
		}

		__device__ ~Material() { delete this; } // Materials are always created dynamically

	private:
		__device__ Material(material_type type) : _type(type) {}
		material_type _type;

		union
		{
			Lambertian _lambertian;
			Metal _metal;
			Dielectric _dielectric;
		};
};

__global__
void create_lambertian(Material* material, vec3 colour)
{
	*material = *Material::lambertian(colour);
}

__global__
void create_metal(Material* material, vec3 colour, float fuzz)
{
	*material = *Material::metal(colour, fuzz);
}

#endif
