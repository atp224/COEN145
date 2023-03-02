Project 1

Introduction
The purpose of this assignment is to become familiar with OpenMP by implementing
dense matrix-matrix multiplication.
Dense matrix multiplication (C = AB)
You need to write a program, matmult_omp, that will take as input two matrices A
and B, and will output their product. A baseline almost complete serial multiplication
program, matmult, is provided to you as C code. It is just an example. You can
write your OpenMP code in C or C++. Complete the program by initializing matrices
with random values and adding a timer solely around the matrix multiplication
function call. You should use the omp_get_wtime function for timing. Your program
should be able to be executed as:
matmult 4 5 3 [optional parameters]
which would create two matrices with random values, A of size 4 x5 and B of size
5 x3 and would multiply them, storing the result into a matrix C of size 4 x3. The
program should print out execution time in microseconds with up to 4 decimal
points. You may optionally add a 4th parameter for nthreads (number of threads the
program should execute the multiplication with).
General program
Design your programs so that they follow the following steps:
1. A single thread creates the matrices.
2. Multiple threads work together to do the matrix multiplication.
3. At the end, one thread reports the execution time.
These are just guidelines, and you should think carefully about step 2. Also, make
sure the matrix multiplication computation is correct.
Parallelization strategy
Implement 2 parallelization strategies: with and without tiling. The goal of the
experiments will be to compare the execution time for the tiling and block (without
tiling) strategies. For the tiling case, you may want to experiment with different tile
sizes to find the optimum values.
