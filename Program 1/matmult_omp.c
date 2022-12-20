/* assert */
    #include <assert.h>

    /* errno */
    #include <errno.h>

    /* fopen, fscanf, fprintf, fclose */
    #include <stdio.h>

    /* EXIT_SUCCESS, EXIT_FAILURE, malloc, free */
    #include <stdlib.h>
    #include <time.h>
    #include <omp.h>


    static int create_mat(size_t const nrows, size_t const ncols, double ** const matp)
    {   
        double * mat=NULL;
        if (!(mat = (double*) malloc(nrows*ncols*sizeof(*mat)))) {
            goto cleanup;
        }
        
        /** Initialize matrix with random values **/
        size_t i, j; 
        for(i = 0; i < nrows; i++){
            for (j = 0; j < ncols; j++){
                mat[(i * ncols) + j] = (double)(rand() % 1000) / 353.0;
            }
        }
        /** End random initialization **/
        
        *matp = mat;
        
        return 0;
        
        cleanup:
        free(mat);
        return -1;
    }

    static void mult_mat(size_t const n, size_t const m, size_t const p,
                    double * A, double *B,
                    double * C)
{ 
  size_t i, j, k;
  double sum;
  
  for (i=0; i<n; ++i) { 
    for (j=0; j<p; ++j) {
      for (k=0, sum=0.0; k<m; ++k) {
        sum += A[i*m+k] * B[k*p+j];
      }
      C[i*p+j] = sum;
    }
  }
}


    static void matrix_matrix_mult_block(double * C, double * A, double * B, int n, int m, int p, int rstart, int rend)
    {
    int r, c, q;
    #pragma omp parallel for collapse(3)
    for (r = rstart; r <= rend; r++)
    {
        for (c = 0; c < m; c++)
        {

        for (q = 0; q < p; q++)
        {
            C[r * m + c] = C[r * m + c] + (A[r * p + q] * B[c * p + q]);
        }
        }
    }
    /*failure:*/
    }

    static void matrix_matrix_mult_by_block(double * C, double * A, double * B, int n, int m, int p, int rtilesize)
    {
    int rstart, rend, cstart, cend, qstart, qend;
    for (rstart = 0; rstart < n; rstart += rtilesize)
    {
        rend = rstart + rtilesize - 1;
        if (rend >= n)
        {
        rend = n-1;
        }
        matrix_matrix_mult_block(C, A , B, n, m, p, rstart, rend);
    }
    }

    static void matrix_matrix_mult_tile(double * C, double * A, double * B, int n, int m, int p, int rstart, int rend, int cstart, int cend, int qstart, int qend)
    {
    int r, c, q;
    #pragma omp parallel for collapse(3)
    for (r = rstart; r <= rend; r++)
    {
        for (c = cstart; c <= cend; c++)
        {

        for (q = qstart; q <= qend; q++)
        {
            C[r * m + c] = C[r * m + c] + (A[r * p + q] * B[c * p + q]);
        }

        }
    }
    }


    static void matrix_matrix_mult_by_tiling(double * C, double * A, double * B, int n, int m, int p, int rtilesize, int ctilesize, int qtilesize)
    {
    size_t rstart, rend, cstart, cend, qstart, qend;
    double sum;
    #pragma omp parallel for 
    for (rstart = 0; rstart < n; rstart += rtilesize)
    {
        rend = rstart + rtilesize - 1;
        if (rend >= n)
        {
        rend = n-1;
        }
        for (cstart = 0; cstart < m; cstart += ctilesize)
            {
            cend = cstart + ctilesize - 1;
                if (cend >= m)
                {
                    cend = m - 1;
                }
            for (qstart = 0; qstart < p; qstart += qtilesize)
            {
                qend = qstart + qtilesize - 1;
                if (qend >= p)
                    {
                    qend = p - 1;
                    }
                matrix_matrix_mult_tile(C, A, B, n, m , p, rstart, rend, cstart, cend, qstart, qend);
            }
        }
    }
    }


    int main(int argc, char * argv[])
    {
        double time_spent = 0.0;
    double begin;
    double end;
    size_t nrows, ncols, ncols2;
    int nthreads;
    int blocksize, tilesize1, tilesize2, tilesize3;
    double * A=NULL, * B=NULL, * C=NULL;


    if (argc != 5) {
        fprintf(stderr, "usage: matmult nrows ncols ncols2 numthreads\n");
        goto failure;
    }

    nrows = atoi(argv[1]);
    ncols = atoi(argv[2]);
    ncols2 = atoi(argv[3]);
    nthreads = atoi(argv[4]);

    if (create_mat(nrows, ncols, &A)) {
        perror("error");
        goto failure;
    }

    if (create_mat(ncols, ncols2, &B)) {
        perror("error");
        goto failure;
    }

   /* if (create_mat(nrows, ncols2, &C)) {
        perror("error");
        goto failure;
    }

    begin = omp_get_wtime();
    mult_mat(nrows,ncols, ncols2, A, B ,C);
    end = omp_get_wtime();
    time_spent = end - begin;
    printf("Serial Time is %f seconds\n" , time_spent);
    free(C);
    */
    if (create_mat(nrows, ncols2, &C)) {
        perror("error");
        goto failure;
    }
    blocksize = nrows/10;
    omp_set_num_threads(nthreads);
    begin = omp_get_wtime();
    matrix_matrix_mult_by_block(C, A, B, nrows, ncols2, ncols, blocksize);
    end = omp_get_wtime();
    time_spent = end - begin;
    printf("Block Time is %f seconds\n" , time_spent);


    free(C);
        if (create_mat(nrows, ncols2, &C)) {
        perror("error");
        goto failure;
    }

    tilesize1 = nrows/10;
    tilesize2 = ncols/10;
    tilesize3 = ncols2/10;

    begin = omp_get_wtime();
    matrix_matrix_mult_by_tiling(C, A, B, nrows, ncols2, ncols, tilesize1, tilesize2, tilesize3);
    end = omp_get_wtime();

    time_spent = end - begin;
    printf("Tiling Time is %f seconds\n" , time_spent);

    free(A);
    free(B);
    free(C);


    return EXIT_SUCCESS;

    failure:
    if(A){
        free(A);
    }
    if(B){
        free(B);
    }
    if(C){
        free(C);
    }
    return EXIT_FAILURE;

    }



