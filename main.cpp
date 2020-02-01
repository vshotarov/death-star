#include "ray.h"
#include "hittable.h"

#include <float.h>
#include <iostream>
#include <fstream>
#include <cstdlib>

#include <glm/glm.hpp>

using namespace glm;


vec3 miss_colour(const ray& r)
{
	vec3 unit_direction = normalize(r.direction);
	float t = .5 * (unit_direction.y + 1.0);
	return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(.5, .7, 1.0);
}

vec3 colour(const ray& r, Hittable* world, int num_hittables)
{
	bool any_hit = false;
	hit_record rec;
	float max_t = MAXFLOAT;

	for(int i=0; i<num_hittables; i++)
	{
		if(world[i].hit(r, .0001, max_t, rec))
		{
			max_t = rec.t;
			any_hit = true;
		}
	}

	if(any_hit)
		return .5f * (rec.normal + 1.0f);

	return miss_colour(r);
}

void render(int width, int height, int num_samples, float* pixel_buffer,
		Hittable* world, int num_hittables)
{
	// NOTE temporarily defining camera variables here before actually
	// implementing camera
	float aspect_ratio = (float)width / float(height);
	vec3 lower_left_corner(-2.0, -1.0, -1.0);
	vec3 horizontal(2.0 * aspect_ratio, .0, .0);
	vec3 vertical(.0, 2.0, .0);
	vec3 origin(.0, .0, .0);

	float f_width = (float)width;
	float f_height = (float)height;
	
	for(int y=0; y<height; y++)
		for(int x=0; x<width; x++)
		{
			// U and V are the 2d coordinates of the camera plane
			float u = float(x) / f_width;
			float v = float(y) / f_height;

			// Get ray through the pixel
			ray r(origin,
				  normalize(lower_left_corner + u*horizontal + v*vertical));

			// Ray trace
			vec3 out_colour = colour(r, world, num_hittables);

			// Store in pixel buffer
			int pixel_id = y * width + x;

			pixel_buffer[pixel_id * 3 + 0] = out_colour.x;
			pixel_buffer[pixel_id * 3 + 1] = out_colour.y;
			pixel_buffer[pixel_id * 3 + 2] = out_colour.z;
		}
}

Hittable* create_world()
{
	Hittable* world;
	world = (Hittable*)malloc(3 * sizeof(Hittable));

	world[0] = Hittable::sphere(vec3(.0, .0, -1.0), .5);
	world[1] = Hittable::triangle(vec3(-1.5f, 0.0f, -1.0f),
								  vec3(-2.0f, 1.0f, -2.0f),
								  vec3(-3.0f, 0.0f, -3.0f));
	world[2] = Hittable::triangle(vec3(.0f, 0.0f, -1.0f),
								  vec3(-.5f, 0.5f, -1.0f),
								  vec3(-1.5f, -.2f, -1.0f));

	return world;
}

int main(int argc, char** argv)
{
	// Parse arguments
	// NOTE this is very erronous at the moment as there is no
	// error catching, validation, etc.
	char *endptr;
	int width = strtol(argv[1], &endptr, 10);
	int height = strtol(argv[2], &endptr, 10);
	int num_samples = strtol(argv[3], &endptr, 10);
	char* out_file = argv[4];

	printf("Initializing death-star for %ix%i pixels and %i samples\n",
			width, height, num_samples);

	// Create scene
	Hittable* world = create_world();

	// Allocate memory for pixels
	float* pixel_buffer;
	pixel_buffer = (float*)malloc(width * height * 3 * sizeof(float));

	// Render into buffer
	render(width, height, num_samples, pixel_buffer, world, 3);

	// Write into ppm file
	std::ofstream out(out_file);
	std::streambuf *coutbuf = std::cout.rdbuf(); // Store old buf
	std::cout.rdbuf(out.rdbuf()); // Redirect cout to out_file

	std::cout<< "P3\n" << width << " " << height << "\n255\n";
	for(int y=height-1; y>=0; y--)
		for(int x=0; x<width; x++)
		{
			int pixel_id = y * width + x;

			int int_r = int(255.99 * pixel_buffer[pixel_id * 3 + 0]);
			int int_g = int(255.99 * pixel_buffer[pixel_id * 3 + 1]);
			int int_b = int(255.99 * pixel_buffer[pixel_id * 3 + 2]);

			std::cout<< int_r <<" "<< int_g << " " << int_b <<std::endl;
		}

	// Restore cout buf
	std::cout.rdbuf(coutbuf);

	return 0;
}
