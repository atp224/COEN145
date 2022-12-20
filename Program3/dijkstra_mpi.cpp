#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <memory.h>

#define ROWMJR(R, C, NR, NC) (R*NC+C)
#define COLMJR(R, C, NR, NC) (C*NR+R)

#define a(R, C) a[ROWMJR(R,C,ln,n)]
#define b(R, C) b[ROWMJR(R,C,nn,n)]

static void findDistance(int ** distance, int ** localMin, int numberOfProcessors, int size){
    int local, reminder;
    if (size < numberOfProcessors){
        local = 1;
        *localMin = (int *) malloc(numberOfProcessors * sizeof(**localMin));
        int i = 0;
        for (; i < size; i++){
            *(*localMin + i) = local;
        }
        for (; i < numberOfProcessors;i++){
            *(*localMin + i) = 0;
        }
    } else {
        local = size / numberOfProcessors;
        reminder = size % numberOfProcessors;

        *localMin = (int *) malloc(numberOfProcessors * sizeof(**localMin));
        for (int i = 0; i < numberOfProcessors; i++) {
            *(*localMin + i) = local;
        }
        *(*localMin + numberOfProcessors - 1) += reminder;
    }
    *distance = (int *) malloc(numberOfProcessors * sizeof(**distance));
    for (int i = 0; i < numberOfProcessors; i++) {
        *(*distance + i) = 0;
        for (int j = 0; j < i; j++) {
            *(*distance + i) += *(*localMin + j);
        }
    }
}
static void
load(
        char const *const filename,
        int *const np,
        float **const ap, int numberOfProcessors, int ** distance, int ** localMin, int rank
) {
    int n;
    float *a = NULL;
    if (rank == 0) {
        int i, j, ret;
        FILE *fp = NULL;


        fp = fopen(filename, "r");
        assert(fp);

        ret = fscanf(fp, "%d", &n);
        assert(1 == ret);

        findDistance(distance, localMin, numberOfProcessors, n);

        a = (float *)malloc(n * *(*localMin) * sizeof(*a));
        for (j = 0; j < *(* localMin) * n; ++j) {
            ret = fscanf(fp, "%f", &a[j]);
            assert(1 == ret);
        }
        *ap = a;
        for (i = 1; i < numberOfProcessors; ++i) {
            a = (float *) malloc(n * *(*localMin + i) * sizeof(*a));
            MPI_Send(&n, 1, MPI_INTEGER, i, 0, MPI_COMM_WORLD);

            MPI_Send(*distance, numberOfProcessors, MPI_INTEGER, i, 1, MPI_COMM_WORLD);
            MPI_Send(*localMin, numberOfProcessors, MPI_INTEGER, i, 2, MPI_COMM_WORLD);
            for (j = 0; j < *(* localMin + i) * n; ++j) {
                ret = fscanf(fp, "%f", &a[j]);
                assert(1 == ret);
            }
            MPI_Send(&j, 1, MPI_INTEGER, i, 4, MPI_COMM_WORLD);
            MPI_Send(a, j, MPI_FLOAT, i, 3, MPI_COMM_WORLD);
            free(a);
        }

        ret = fclose(fp);
        assert(!ret);
    } else {
        int count;
        MPI_Recv(&n, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        *distance = (int *) malloc(numberOfProcessors * sizeof(**distance));
        MPI_Recv(*distance, numberOfProcessors, MPI_INTEGER, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        *localMin = (int *) malloc(numberOfProcessors * sizeof(**localMin));
        MPI_Recv(*localMin, numberOfProcessors, MPI_INTEGER, 0, 2, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(&count, 1, MPI_INTEGER, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        a = (float*) malloc(count * sizeof(*a));
        MPI_Recv(a, count, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        *ap = a;
    }

    *np = n;
    MPI_Barrier(MPI_COMM_WORLD);
}

static void
dijkstra(
        int const source,
        int const n,
        float const *const a,
        float **const result, int rank, int * distance, int * localMin, int numberOfProcessors
) {
    int i, j, sourceNode = 0;
    struct float_int {
        float distance;
        int u;
    } min;
    char *set = NULL;
    float *resultVector = NULL;
    float * localResult = NULL; 

    set = (char*) calloc(n, sizeof(*set));
    assert(set);

    resultVector = (float*) malloc(n * sizeof(*resultVector));
    assert(resultVector);

    localResult = (float*) malloc(n * sizeof(*resultVector));
    assert(localResult);

    for (i = 0; i < numberOfProcessors; i++){
        if (source < distance[i]){
            sourceNode = i - 1;
            break;
        }
    }
    
    if (rank == sourceNode) {
        for (i = 0; i < n; ++i) {
            resultVector[i] = a[i + n * (source - distance[sourceNode])];
            
        }
    }
    MPI_Bcast(resultVector, n, MPI_FLOAT, sourceNode, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    set[source] = 1;
    min.u = -1; 

    for (i = 1; i < n; ++i) {
        min.distance = INFINITY;
        for (j = 0; j < n; ++j) {
            if (!set[j] && resultVector[j] < min.distance) {
                min.distance = resultVector[j];
                min.u = j;
            }
            localResult[j] = resultVector[j];
        }
       
        set[min.u] = 1;
        for (j = 0; j < localMin[rank]; j++){
            if (set[j + distance[rank]]){
                continue;
            }
            if (a(j, min.u) + min.distance < localResult[j + distance[rank]]){
                localResult[j + distance[rank]] = a(j, min.u) + min.distance;
            }
        }

        MPI_Allreduce(localResult, resultVector, n, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    free(set);

    *result = resultVector;
}

static void
print_time(double const seconds,int const processor) {
    printf("Operation Time for %d processors: %0.04fs\n",processor, seconds);
}

static void
print_numbers(
        char const *const filename,
        int const n,
        float const *const numbers) {
    int i;
    FILE *fout;

    if (NULL == (fout = fopen(filename, "w"))) {
        fprintf(stderr, "error opening '%s'\n", filename);
        abort();
    }

    for (i = 0; i < n; ++i) {
        fprintf(fout, "%10.4f\n", numbers[i]);
    }

    fclose(fout);
}

int
main(int argc, char **argv) {
    int n, numberOfProcessors, rank;;
    double ts, te;
    float *a = NULL, *result = NULL;
    int * distance = NULL, *localMin = NULL;

    if (argc < 4) {
        printf("Invalid number of arguments.\nUsage: dijkstra <graph> <source> <output_file>.\n");
        return EXIT_FAILURE;
    }
    srand(time(NULL));
    unsigned int seed = time(NULL);

    int nsources = atoi(argv[2]);
    assert(nsources > 0);

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        printf("Loading graph from %s.\n", argv[1]);
    }
    load(argv[1], &n, &a, numberOfProcessors, &distance, &localMin, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0)
    {
    printf("Performing %d searches from random sources.\n", nsources);
    }
    ts = MPI_Wtime();
    for(int i=0; i < nsources; ++i){
        dijkstra(rand_r(&seed) % n, n, a, &result, rank, distance, localMin, numberOfProcessors);
    }
    te = MPI_Wtime();

    if (rank == 0) {
        print_time((te - ts)/numberOfProcessors,numberOfProcessors);
        print_numbers(argv[3], n, result);
    }
    free(a);
    free(result);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
