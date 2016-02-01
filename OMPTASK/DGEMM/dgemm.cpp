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

NAME:    dgemm

PURPOSE: This program tests the efficiency with which a dense matrix
dense multiplication is carried out

USAGE:   The program takes as input the number of threads, the matrix
order, the number of times the matrix-matrix multiplication 
is carried out, and, optionally, a tile size for matrix
blocking

<progname> <# threads> <# iterations> <matrix order> [<tile size>]

The output consists of diagnostics to make sure the 
algorithm worked, and of timing statistics.

FUNCTIONS CALLED:

Other than OpenMP or standard C functions, the following 
functions are used in this program:

wtime()
bail_out()

HISTORY: Written by Rob Van der Wijngaart, September 2006.
Made array dimensioning dynamic, October 2007
Allowed arbitrary block size, November 2007
Removed reverse-engineered MKL source code option, November 2007
Changed from row- to column-major storage order, November 2007
Stored blocks of B in transpose form, November 2007

 ***********************************************************************************/

#include <par-res-kern_general.h>
#include <par-res-kern_omp.h>

#ifndef DEFAULTBLOCK
#define DEFAULTBLOCK 32
#endif

#ifndef BOFFSET
#define BOFFSET 12
#endif
#define AA_arr(i,j) AA[(i)+(block+BOFFSET)*(j)]
#define BB_arr(i,j) BB[(i)+(block+BOFFSET)*(j)]
#define CC_arr(i,j) CC[(i)+(block+BOFFSET)*(j)]
#define  A_arr(i,j)  A[(i)+(order)*(j)]
#define  B_arr(i,j)  B[(i)+(order)*(j)]
#define  C_arr(i,j)  C[(i)+(order)*(j)]

#define forder (1.0*order)

main(int argc, char **argv) {

    int     iterations;           /* number of times the multiplication is done     */
    double  dgemm_time,           /* timing parameters                              */
            avgtime;
    double  checksum = 0.0,       /* checksum of result                             */
            ref_checksum;
    double  epsilon = 1.e-8;      /* error tolerance                                */
    int     nthread_input,        /* thread parameters                              */
            nthread;   
    long    order;                /* number of rows and columns of matrices         */
    int     block;                /* tile size of matrices                          */

    printf("Parallel Research Kernels version %s\n", PRKVERSION);
    printf("OpenMP Dense matrix-matrix multiplication\n");

    if (argc != 4 && argc != 5) {
        printf("Usage: %s <# threads> <# iterations> <matrix order> [tile size]\n",*argv);
        exit(EXIT_FAILURE);
    }

    nthread_input = atoi(*++argv); 

    if ((nthread_input < 1) || (nthread_input > MAX_THREADS)) {
        printf("ERROR: Invalid number of threads: %d\n", nthread_input);
        exit(EXIT_FAILURE);
    }

    omp_set_num_threads(nthread_input);

    iterations = atoi(*++argv);
    if (iterations < 1){
        printf("ERROR: Iterations must be positive : %d \n", iterations);
        exit(EXIT_FAILURE);
    }

    order = atol(*++argv);
    if (order < 1) {
        printf("ERROR: Matrix order must be positive: %ld\n", order);
        exit(EXIT_FAILURE);
    }

    if (argc == 5) {
        block = atoi(*++argv);
    } else block = DEFAULTBLOCK;

    double *A1 = new double[order*order];
    double *B1 = new double[order*order];
    double *C1 = new double[order*order];

    double *A2 = new double[order*order];
    double *B2 = new double[order*order];
    double *C2 = new double[order*order];
    if (!A1 || !B1 || !C1 || !A2 || !B2 || !C2) {
        printf("ERROR: Could not allocate space for global matrices\n");
        exit(EXIT_FAILURE);
    }

    double *A = A1, *B = B1, *C = C1;
    double *A_next = A2, *B_next = B2, *C_next = C2;

    ref_checksum = (0.25*forder*forder*forder*(forder-1.0)*(forder-1.0));

#pragma omp parallel
    {
#pragma omp for 
        for(int j = 0; j < order; j++) {
            for(int i = 0; i < order; i++) {
                A_arr(i,j) = (double) j;
                B_arr(i,j) = (double) j; 
                C_arr(i,j) = 0.0;
            }
        }
#pragma omp master 
        {
            nthread = omp_get_num_threads();
            printf("Matrix order                = %ld\n", order);
            printf("Number of threads requested = %d\n", nthread_input);
            printf("Number of threads received  = %d\n", nthread);
            printf("Blocking factor             = %d\n", block);
            printf("Number of iterations        = %d\n", iterations);

            for (int iter=0; iter<=iterations; iter++) {
                if (iter==1) {
                    printf("starting timer\n");
                    dgemm_time = wtime();
                }
                for(int jj = 0; jj < order; jj+=block) {
                    for(int kk = 0; kk < order; kk+=block) {
#pragma omp task depend(out: BB_arr(jj,kk))
                        for (int jg=jj,j=0; jg<MIN(jj+block,order); j++,jg++) {
                            for (int kg=kk,k=0; kg<MIN(kk+block,order); k++,kg++) {
                                BB_arr(j,k) =  B_arr(kg,jg);
                            }
                        }
                        for(int ii = 0; ii < order; ii+=block) {
#pragma omp task depend(out: AA_arr(ii,kk))
                            for (int kg=kk,k=0; kg<MIN(kk+block,order); k++,kg++) {
                                for (int ig=ii,i=0; ig<MIN(ii+block,order); i++,ig++) {
                                    AA_arr(i,k) = A_arr(ig,kg);
                                }
                            }
#pragma omp task depend(out: CC_arr(0,0))
                            for (int jg=jj,j=0; jg<MIN(jj+block,order); j++,jg++) {
                                for (int ig=ii,i=0; ig<MIN(ii+block,order); i++,ig++) {
                                    CC_arr(i,j) = 0.0;
                                }
                            }
#pragma omp task depend(out: CC_arr(0,0)) depend(in: AA_arr(0,0), BB_arr(0,0), CC_arr(0,0)) 
                            for (int kg=kk,k=0; kg<MIN(kk+block,order); k++,kg++) {
                                for (int jg=jj,j=0; jg<MIN(jj+block,order); j++,jg++)  {
                                    for (int ig=ii,i=0; ig<MIN(ii+block,order); i++,ig++) {
                                        CC_arr(i,j) += AA_arr(i,k)*BB_arr(j,k);
                                    }
                                }
                            }
#pragma omp task depend(out: C_arr(ii,jj)) depend(in: CC_arr(0,0)) 
                            for (int jg=jj,j=0; jg<MIN(jj+block,order); j++,jg++) {
                                for (int ig=ii,i=0; ig<MIN(ii+block,order); i++,ig++) {
                                    C_arr(ig,jg) += CC_arr(i,j);
                                }
                            }
                        }
                    }  
                }
#pragma omp taskwait
/*
#pragma omp for 
                for (int jg=0; jg<order; jg++) {
                    for (int kg=0; kg<order; kg++) {
                        for (int ig=0; ig<order; ig++) {
                            C_arr(ig,jg) += A_arr(ig,kg)*B_arr(kg,jg);
                        }
                    }
                }
*/
            } // end of iterations
            dgemm_time = wtime() - dgemm_time;
        }//end of Master
    } // end of parallel region

    checksum = 0.0;
    for(int j = 0; j < order; j++) {
        for(int i = 0; i < order; i++) {
            checksum += C_arr(i,j);
        }
    }

    // verification test 
    ref_checksum *= (iterations+1);

    if (ABS((checksum - ref_checksum)/ref_checksum) > epsilon) {
        printf("ERROR: Checksum = %lf, Reference checksum = %lf\n",
                checksum, ref_checksum);
        exit(EXIT_FAILURE);
    } else {
        printf("Solution validates\n");
#ifdef VERBOSE
        printf("Reference checksum = %lf, checksum = %lf\n", 
                ref_checksum, checksum);
#endif
    }

    double nflops = 2.0*forder*forder*forder;
    avgtime = dgemm_time/iterations;
    printf("Rate (MFlops/s): %lf  Avg time (s): %lf\n", 1.0E-06 *nflops/avgtime, avgtime);

    //exit(EXIT_SUCCESS);
    return 0;
}
