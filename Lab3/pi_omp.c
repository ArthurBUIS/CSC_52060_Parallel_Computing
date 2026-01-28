/**
 * INF560 - TD4
 *
 * Part 2-a
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int
main( int argc, char ** argv )
{
	int N ;
	int seed ;
	int i ;
	int m = 0 ;
	double pi ;
	double time_start, time_stop, duration ;

	/* Check the number of command-line arguments */
	if ( argc != 3 )
	{
		fprintf( stderr, "Usage: %s N seed\n", argv[0]);
		return EXIT_FAILURE;
	}

	/* Grab the input parameters from command line */
	N = atoi(argv[1]);
	seed = atoi(argv[2]);

	/* Check input-parameter values */
	if ( N <= 0 ) 
	{
		fprintf( stderr, "Error: N should be positive\n" ) ;
		return EXIT_FAILURE ;
	}

	/* Star timer */
	time_start = omp_get_wtime() ;

    printf( "Running w/ N=%d, initial seed=%d\n", N, seed );

    #pragma omp parallel 
    {
        struct drand48_data randBuffer;
        int p = omp_get_thread_num();

        // Each thread gets a unique seed!!
        srand48_r(seed + p, &randBuffer);

        printf( "Thread %d running with seed %d\n", p, seed + p );

        #pragma omp for
	    for( i = 0 ; i < N ; i++ ) 
        {

            double x_rand, y_rand;
            double x, y;

            drand48_r(&randBuffer, &x_rand);
            drand48_r(&randBuffer, &y_rand);

            x = 1 - (2 * x_rand);
            y = 1 - (2 * y_rand);

            if((x*x + y*y) < 1) 
            {
    #if DEBUG
                printf("x=%lf, y=%f is IN\n", x, y);
    #endif
                #pragma omp atomic
                m++;
            } else {
    #if DEBUG
                printf("x=%lf, y=%f is OUT\n", x, y);
    #endif
            }
	    }
    }

	/* Stop timer */
	time_stop = omp_get_wtime() ;

    /* The code ended up finding pi with 5 significant digits for N = 1,000,000,000, with 
    less than 30s execution time: not bad!! */

#if DEBUG
  printf("m=%d\n", m);
#endif

  /* Compute value of PI */
  pi = (double)4*m/N;

  printf("Result -> PI = %f\n", pi);

  /* Compute final duration (in seconds) */
  duration = time_stop - time_start ;

  printf("Computed in %g s\n", duration);

  return EXIT_SUCCESS;
}