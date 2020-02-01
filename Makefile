CC=g++
CFLAGS=-I. -I./dep/include -L./dep/lib

main: main.cpp
	$(CC) -o run main.cpp $(CFLAGS)

.PHONY: clean

clean:
	rm run
