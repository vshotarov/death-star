#include <iostream>
#include <fstream>
#include <cstdlib>

#include <glm/glm.hpp>


void render(int width, int height, int num_samples, float* pixel_buffer)
{
	for(int y=0; y<height; y++)
		for(int x=0; x<width; x++)
		{
			int pixel_id = y * width + x;

			pixel_buffer[pixel_id * 3 + 0] = .0f;
			pixel_buffer[pixel_id * 3 + 1] = y / (float)height;
			pixel_buffer[pixel_id * 3 + 2] = x / (float)width;
		}
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

	// Allocate memory for pixels
	float* pixel_buffer;
	pixel_buffer = (float*)malloc(width * height * 3 * sizeof(float));

	// Render into buffer
	render(width, height, num_samples, pixel_buffer);

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
