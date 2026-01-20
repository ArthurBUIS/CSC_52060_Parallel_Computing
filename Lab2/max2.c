#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char**argv) {
  int rank, size;
  int s ;
  int n ;
  int * tab ;
  int i;
  int max ;
  double t1, t2;

  /* MPI Initialization */
  MPI_Init(&argc, &argv);

  /* Get the rank of the current task and the number
   * of MPI processes
   */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* Check the input arguments */
  if(argc <3) {
    printf("Usage: %s S N\n", argv[0]);
    printf( "\tS: seed for pseudo-random generator\n" ) ;
    printf( "\tN: size of the array\n" ) ;
    exit( 1 ) ;
  }

  s = atoi(argv[1]);
  n = atoi(argv[2]);
  srand48(s);

  /* Allocate the array */
  tab = malloc(sizeof(int) * n);
  if ( tab == NULL ) { 
	  fprintf( stderr, "Unable to allocate %d elements\n", n ) ;
	  return 1 ; 
  }

  /* Initialize the array for rank 0 only !! */
  if (rank == 0) {
    for(i=0; i<n; i++) {
      tab[i] = lrand48()%n;
    }
  }

  /* start the measurement */
  t1=MPI_Wtime();

  int n_per_proc = n / size;
  int start = rank * n_per_proc;
  int end = (rank == size - 1) ? n : start + n_per_proc;

  /* search for the max value */
  int local_max = tab[start];
  for(i=start; i<end; i++) {
    if(tab[i] > local_max) {
      local_max = tab[i];
    }
  }

  int global_max = local_max;
  if(rank == 0) {
    int recv_max;
      for(i = 1; i < size; i++) {
        MPI_Recv(&recv_max, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if(recv_max > global_max) global_max = recv_max;
    }
  } else {
    MPI_Send(&local_max, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

// Doit marcher avec ce code aussi !! Mais j'ai supposé que cette fonction ne faisait pas partie du cours pour le moment
//   int global_max;
//   MPI_Reduce(&local_max, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  /* stop the measurement */
  t2=MPI_Wtime();

  if (rank == 0) {
    printf("Computation time: %f s\n", t2-t1);
  }

#if DEBUG
  printf("the array contains:\n");
  for(i=0; i<n; i++) {
    printf("%d  ", tab[i]);
  }
  printf("\n");
#endif

  if (rank == 0) {
    printf("(Seed %d, Size %d) Max value = %d, Time = %g s\n", s, n, global_max, t2-t1);      
  }
  MPI_Finalize();
  return 0;
}