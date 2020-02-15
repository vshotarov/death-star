CC=nvcc
CFLAGS=-I. -I./dep/include -L./dep/lib -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -G

main: main.cu
	$(CC) -o run main.cu $(CFLAGS)

.PHONY: clean

clean:
	rm run
