CC = g++

default: sparsematmult

sparsematmult: sparsematmult.cpp
	${CC} -O0 -g -Wall -Wextra -Wno-unused-parameter -fopenmp -o $@ sparsematmult.cpp -std=c++11

clean:
	-rm -f sparsematmult

