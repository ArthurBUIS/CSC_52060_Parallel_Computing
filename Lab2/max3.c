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

    if (rank == 0) {
        srand48(s);
        tab = malloc(n * sizeof(int));
    } else {
        tab = malloc(n * sizeof(int));
    }

    t1 = MPI_Wtime();

    for (j = 0; j < m; j++) {

        if (rank == 0) {
            for (i = 0; i < n; i++)
                tab[i] = lrand48() % n;
        }

        MPI_Bcast(tab, n, MPI_INT, 0, MPI_COMM_WORLD);

        int chunk = n / size;
        int start = rank * chunk;
        int end = (rank == size - 1) ? n : start + chunk;

        local_max = tab[start];
        for (i = start; i < end; i++)
            if (tab[i] > local_max)
                local_max = tab[i];

        // Cette fois je me permet l'utilisation de MPI_Reduce, car j'ai déjà implémenté un 
        // équivalent dans max2.c
        MPI_Reduce(&local_max, &global_max, 1, MPI_INT,
                   MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0)
            printf("Array %d: max = %d\n", j, global_max);
    }

    t2 = MPI_Wtime();

    if (rank == 0)
        printf("Time: %f s\n", t2 - t1);

    free(tab);
    MPI_Finalize();
    return 0;
}
