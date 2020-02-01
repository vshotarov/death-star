#include <iostream>
#include <cstdlib>

#include <glm/glm.hpp>


int main(int argc, char** argv)
{
	// Parse arguments
	// NOTE this is very erronous at the moment as there is no
	// error catching, validation, etc.
	char *endptr;
	int width = strtol(argv[1], &endptr, 10);
	int height = strtol(argv[2], &endptr, 10);
	int num_samples = strtol(argv[3], &endptr, 10);

	printf("Initializing death-star for %ix%i pixels and %i samples\n",
			width, height, num_samples);

	return 0;
}
