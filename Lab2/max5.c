#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int rank, size;
    int s;
    int n;
    int m;
    int *tab = NULL;
    int i, j;
    int local_max, global_max;
    double t1, t2;

    /* MPI Initialization */
    MPI_Init(&argc, &argv);
    /* Get the rank of the current task and the number
    * of MPI processes
    */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Check the input arguments */
    if (argc < 4) {
        if (rank == 0)
            printf("Usage: %s S N M\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    s = atoi(argv[1]);
    n = atoi(argv[2]);
    m = atoi(argv[3]);

    tab = malloc(n * sizeof(int));

    if (rank == 0) {
        srand48(s);

        int active = size - 1;
        MPI_Status status;

        while (active > 0) {
            int worker;
            MPI_Recv(&worker, 1, MPI_INT, MPI_ANY_SOURCE,
                     MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (next_array < m) {
                for (i = 0; i < n; i++)
                    tab[i] = lrand48() % n;

                MPI_Send(tab, n, MPI_INT, worker, WORK, MPI_COMM_WORLD);
                next_array++;
            } else {
                MPI_Send(NULL, 0, MPI_INT, worker, STOP, MPI_COMM_WORLD);
                active--;
            }
        }
    }
    else {
        MPI_Status status;

        while (1) {
            MPI_Send(&rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(tab, n, MPI_INT, 0, MPI_ANY_TAG,
                     MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == STOP)
                break;

            max = tab[0];
            for (i = 1; i < n; i++)
                if (tab[i] > max)
                    max = tab[i];

            printf("Rank %d max = %d\n", rank, max);
        }
    }

    free(tab);
    MPI_Finalize();
    return 0;
}
