#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Fonction auxiliaire de add_vectors
void __global__ add_vectors_kernel( int * a, int * b, int *c, int N) 
{
int i = blockIdx.x * blockDim.x + threadIdx.x;

if ( i < N ) 
{
c[i] = a[i] + b[i] ; 
}
}

/* Function computing the final string to print */
void add_vectors( int * a, int * b, int *c, int N ) 
{

// En reprenant l'exple de cours
// 1. Device pointers
int * d_a, * d_b, * d_c;

// 2. Memory allocation on GPU
cudaMalloc( (void**)&d_a, N * sizeof(int) ) ;
cudaMalloc( (void**)&d_b, N * sizeof(int) ) ;
cudaMalloc( (void**)&d_c, N * sizeof(int) ) ;

// 3. Transfer to GPU
cudaMemcpy( d_a, a, N * sizeof(int), cudaMemcpyHostToDevice ) ;
cudaMemcpy( d_b, b, N * sizeof(int), cudaMemcpyHostToDevice ) ;
cudaMemcpy( d_c, c, N * sizeof(int), cudaMemcpyHostToDevice ) ;

// 4. Kernel execution
    // On prend des blocs de 256 threads
    int threadsPerBlock = 256;
    // On calcule le nombre de blocs nécessaires pour couvrir N
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Lancement du Kernel
    add_vectors_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

// 5. Transfer to host
cudaMemcpy( c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost ) ;

// 6. Free GPU memory
cudaFree( d_a ) ;
cudaFree( d_b ) ;
cudaFree( d_c ) ;
}

int
main( int argc, char ** argv )
{
	int N ;
	int seed ;

    /* Check the number of command-line arguments */
	if ( argc != 3 )
	{
		fprintf( stderr, "Usage: %s N seed\n", argv[0]);
		return EXIT_FAILURE;
	}

	/* Grab the input parameters from command line */
    N = atoi( argv[1] ) ;
    seed = atoi( argv[2] ) ;

	srand48(seed);

    int* a = (int*) malloc( N * sizeof(int) ) ;
    int* b = (int*) malloc( N * sizeof(int) ) ;
    int* c = (int*) malloc( N * sizeof(int) ) ;

    // Initialisation des tableaux

    for ( int i = 0 ; i < N ; i++ )
    {
        a[i] = lrand48() % 100 ;
        b[i] = lrand48() % 100 ;
        c[i] = 0 ;
    }

    struct timeval start, end;

    gettimeofday(&start, NULL); // Top chrono

    add_vectors(a, b, c, N);

    gettimeofday(&end, NULL); // Fin chrono

    // Vérification du résultat (en séquentiel)
    for ( int i = 0 ; i < N ; i++ )
    {
        if ( c[i] != a[i] + b[i] )
        {
            fprintf( stderr, "Erreur à l'index %d: expected %d, got %d\n", i, a[i] + b[i], c[i] );
            return EXIT_FAILURE;
        }
    }

    printf("Tout est correct!\n");

    double duration = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("Temps total (Transferts + Kernel): %f secondes\n", duration);
    return 0 ;
}