/*
   Copyright (c) 2013, Intel Corporation

   Redistribution and use in source and binary forms, with or without 
   modification, are permitted provided that the following conditions 
   are met:

 * Redistributions of source code must retain the above copyright 
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above 
 copyright notice, this list of conditions and the following 
 disclaimer in the documentation and/or other materials provided 
 with the distribution.
 * Neither the name of Intel Corporation nor the names of its 
 contributors may be used to endorse or promote products 
 derived from this software without specific prior written 
 permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
 FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
 COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
 ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 POSSIBILITY OF SUCH DAMAGE.
 */

/*********************************************************************************

NAME:    sparse

PURPOSE: This program tests the efficiency with which a sparse matrix
vector multiplication is carried out

USAGE:   The program takes as input the number of threads, the 2log of the linear
size of the 2D grid (equalling the 2log of the square root of the order
of the sparse matrix), the radius of the difference stencil, and the number 
of times the matrix-vector multiplication is carried out.

<progname> <# threads> <# iterations> <2log root-of-matrix-order> <radius> 

The output consists of diagnostics to make sure the 
algorithm worked, and of timing statistics.

FUNCTIONS CALLED:

Other than OpenMP or standard C functions, the following 
functions are used in this program:

wtime()
bail_out()
reverse()

NOTES:   

HISTORY: Written by Rob Van der Wijngaart, August 2006.
Updated by RvdW to parallelize matrix generation, March 2007.
Updated by RvdW to fix verification bug, February 2013
Updated by RvdW to sort matrix elements to reflect traditional CSR storage,
August 2013

 ***********************************************************************************/

#include <par-res-kern_general.h>
#include <par-res-kern_omp.h>

/* linearize the grid index                                                       */
#define LIN(i,j) (i+((j)<<input_size))

/* if the scramble flag is set, convert all (linearized) grid indices by 
   reversing their bits; if not, leave the grid indices alone                     */
#ifdef SCRAMBLE
#define REVERSE(a,b)  reverse((a),(b))
#else
#define REVERSE(a,b) (a)
#endif

#define BITS_IN_BYTE 8

static u64Int reverse(register u64Int, int);
static int compare(const void *el1, const void *el2);

void check_input(int nthread_input, int iterations, int input_size, int radius) {
    if ((nthread_input < 1) || (nthread_input > MAX_THREADS)) {
        printf("ERROR: Invalid number of threads: %d\n", nthread_input);
        exit(EXIT_FAILURE);
    }

    if (iterations < 1){
        printf("ERROR: Iterations must be positive : %d \n", iterations);
        exit(EXIT_FAILURE);
    }

    if (input_size <0) {
        printf("ERROR: Log of grid size must be greater than or equal to zero: %d\n", 
                (int) input_size);
        exit(EXIT_FAILURE);
    }

    //emit error if (periodic) stencil overlaps with itself
    
    if ((1<<input_size) < 2*radius+1) {
        printf("ERROR: Grid extent %d smaller than stencil diameter 2*%d+1= %d\n",
                (1<<input_size), radius, radius*2+1);
        exit(EXIT_FAILURE);
    }

    if (radius <0) {
        printf("ERROR: Stencil radius must be non-negative: %d\n", (int) radius);
        exit(EXIT_FAILURE);
    }

}

int main(int argc, char **argv){

    int               input_size; /* logarithmic linear size of grid                */
    int               log_size;   /* logarithmic size of grid                       */
    int               size;       /* linear size of grid                            */
    int               radius,     /* stencil parameters                             */
                      stencil_size; 
    int               iterations; /* number of times the multiplication is done     */
    double            sparsity;   /* fraction of non-zeroes in matrix               */
    double            sparse_time,/* timing parameters                              */
                      avgtime;
    double            vector_sum; /* checksum of result                             */
    double            reference_sum; /* checksum of "rhs"                           */
    double            epsilon = 1.e-8; /* error tolerance                           */
    int               nthread_input,  /* thread parameters                          */
                      nthread;   
    int               num_error=0; /* flag that signals that requested and 
                                      obtained numbers of threads are the same       */

    printf("Parallel Research Kernels version %s\n", PRKVERSION);
    printf("OpenMP Sparse matrix-vector multiplication\n");

    if (argc != 5) {
        printf("Usage: %s <# threads> <# iterations> <2log grid size> <stencil radius>\n",*argv);
        exit(EXIT_FAILURE);
    }

    nthread_input = atoi(argv[1]); 
    iterations    = atoi(argv[2]);
    input_size    = atoi(argv[3]);
    radius        = atoi(argv[4]);

    check_input(nthread_input, iterations, input_size, radius);

    omp_set_num_threads(nthread_input);

    log_size = 2*input_size;
    size = 1<<input_size;
    //compute number of points in the grid
    int64_t grid_size = size*size;

    //compute total size of star stencil in 2D
    stencil_size = 4*radius+1;
    //sparsity follows from number of non-zeroes per row
    sparsity = (double)(4*radius+1)/(double)grid_size;

    //compute total number of non-zeroes
    int64_t num_entries = grid_size*stencil_size;

    double  * RESTRICT matrix   = new double[num_entries];
    double  * RESTRICT vector   = new double[grid_size];
    double  * RESTRICT result   = new double[grid_size];
    int64_t * RESTRICT colIndex = new int64_t[num_entries];

#pragma omp parallel
    {
#pragma omp master 
        {
            nthread = omp_get_num_threads();
            if (nthread != nthread_input) {
                num_error = 1;
                printf("ERROR: number of requested threads %d does not equal ",
                        nthread_input);
                printf("number of spawned threads %d\n", nthread);
            } 
            else {
                printf("Number of threads     = %16d\n",nthread_input);
                printf("Matrix order          = "FSTR64U"\n", grid_size);
                printf("Stencil diameter      = %16d\n", 2*radius+1);
                printf("Sparsity              = %16.10lf\n", sparsity);
                printf("Number of iterations  = %16d\n", iterations);
#ifdef SCRAMBLE
                printf("Using scrambled indexing\n");
#else
                printf("Using canonical indexing\n");
#endif
            }
        }
        bail_out(num_error);

        /* initialize the input and result vectors                                      */
#pragma omp for
        for (int row=0; row<grid_size; row++) {
            result[row] = vector[row] = 0.0;
        }

        /* fill matrix with nonzeroes corresponding to difference stencil. We use the 
           scrambling for reordering the points in the grid.                            */

#pragma omp for
        for (int row=0; row<grid_size; row++) {
            int64_t j = row/size; 
            int64_t i = row%size;
            int64_t elm = row*stencil_size;
            //colIndex[elm] = REVERSE(LIN(i,j),log_size);
            colIndex[elm] = i+j*size;
            for (int r=1; r<=radius; r++, elm+=4) {
                //colIndex[elm+1] = REVERSE(LIN((i+r     )%size, j             ),log_size);
                //colIndex[elm+2] = REVERSE(LIN((i-r+size)%size, j             ),log_size);
                //colIndex[elm+3] = REVERSE(LIN( i             ,(j+r)%size     ),log_size);
                //colIndex[elm+4] = REVERSE(LIN( i             ,(j-r+size)%size),log_size);
                
                colIndex[elm+1] = (i+r     )%size + j                *size;
                colIndex[elm+2] = (i-r+size)%size + j                *size;
                colIndex[elm+3] =  i              + ((j+r     )%size)*size;
                colIndex[elm+4] =  i              + ((j-r+size)%size)*size;
            }
            /* sort colIndex to make sure the compressed row accesses
               vector elements in increasing order                                         */
            qsort(&(colIndex[row*stencil_size]), stencil_size, sizeof(int64_t), compare);
            for (elm=row*stencil_size; elm<(row+1)*stencil_size; elm++) {
                matrix[elm] = 1.0/(double)(colIndex[elm]+1);
            }
        }

        for (int iter=0; iter<=iterations; iter++) {
            /* start timer after a warmup iteration                                        */
            if (iter == 1) { 
#pragma omp barrier
#pragma omp master
                {   
                    sparse_time = wtime();
                }
            }
            // fill vector - Why is populating vector in the timed part?
#pragma omp for 
            for (int row=0; row<grid_size; row++) {
                vector[row] += (double) (row+1);
            }
            // do the actual matrix-vector multiplication 
#pragma omp for
            for (int64_t row=0; row<grid_size; row++) {
                //int64_t first = stencil_size * row; 
                //int64_t last  = first + stencil_size-1;
//#pragma simd reduction(+:temp) 
                    //for (temp=0.0, col=first; col<=last; col++) { temp += matrix[col]*vector[colIndex[col]];
                    //the number of entries in each row in a real sparse MV mult wouldn't be known.
                    for (int64_t col=0; col<stencil_size; col++) {
                        result[row] += matrix[col + stencil_size*row] * vector[colIndex[col + stencil_size*row]];
                    }
                    //result[row] += temp;
                }
            } /* end of iterations                                                          */

#pragma omp barrier
#pragma omp master
            {
                sparse_time = wtime() - sparse_time;
            }

        } /* end of parallel region                                                     */

        /* verification test                                                            */
        reference_sum = 0.5 * (double) num_entries * (double) (iterations+1) * 
            (double) (iterations +2);

        vector_sum = 0.0;
        for (int row=0; row<grid_size; row++) {
            vector_sum += result[row];
        }
        if (ABS(vector_sum-reference_sum) > epsilon) {
            printf("ERROR: Vector sum = %lf, Reference vector sum = %lf\n",
                    vector_sum, reference_sum);
            exit(EXIT_FAILURE);
        }
        else {
            printf("Solution validates\n");
#ifdef VERBOSE
            printf("Reference sum = %lf, vector sum = %lf\n", 
                    reference_sum, vector_sum);
#endif
        }

        avgtime = sparse_time/iterations;
        printf("Rate (MFlops/s): %lf  Avg time (s): %lf\n",
                1.0E-06 * (2.0*num_entries)/avgtime, avgtime);

        exit(EXIT_SUCCESS);
    }

    /* Code below reverses bits in unsigned integer stored in a 64-bit word.
       Bit reversal is with respect to the largest integer that is going to be
       processed for the particular run of the code, to make sure the reversal
       constitutes a true permutation. Hence, the final result needs to be shifted 
       to the right.
    Example: if largest integer being processed is 0x000000ff = 255 = 
    0000...0011111111 (binary), then the unshifted reversal of 0x00000006 = 6 =
    0000...0000000110 (binary) would be 011000000...0000 = 3*2^61, which is 
    outside the range of the original sequence 0-255. Setting shift_in_bits to
    2log(256) = 8, the final result is shifted the the right by 64-8=56 bits,
    so we get 000...0001100000 (binary) = 96, which is within the proper range */
    u64Int reverse(register u64Int x, int shift_in_bits){ 
        x = ((x >> 1)  & 0x5555555555555555) | ((x << 1)  & 0xaaaaaaaaaaaaaaaa);
        x = ((x >> 2)  & 0x3333333333333333) | ((x << 2)  & 0xcccccccccccccccc);
        x = ((x >> 4)  & 0x0f0f0f0f0f0f0f0f) | ((x << 4)  & 0xf0f0f0f0f0f0f0f0);
        x = ((x >> 8)  & 0x00ff00ff00ff00ff) | ((x << 8)  & 0xff00ff00ff00ff00);
        x = ((x >> 16) & 0x0000ffff0000ffff) | ((x << 16) & 0xffff0000ffff0000);
        x = ((x >> 32) & 0x00000000ffffffff) | ((x << 32) & 0xffffffff00000000);
        return (x>>((sizeof(u64Int)*BITS_IN_BYTE-shift_in_bits)));
    }

    int compare(const void *el1, const void *el2) {
        int64_t v1 = *(int64_t *)el1;  
        int64_t v2 = *(int64_t *)el2;
        return (v1<v2) ? -1 : (v1>v2) ? 1 : 0;
    }
