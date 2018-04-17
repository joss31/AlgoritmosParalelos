#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

const int RMAX = 100;

void Get_input(int argc, char* argv[], int* global_n_p, int* local_n_p,
               int my_rank, int p, MPI_Comm comm) {
    
    if (my_rank == 0) {
        if (argc != 3) {
            *global_n_p = -1;
        } else {
            
            *global_n_p = strtol(argv[2], NULL, 10);
            if (*global_n_p % p != 0) {
                *global_n_p = -1;
            }
            
        }
    }
    
    MPI_Bcast(global_n_p, 1, MPI_INT, 0, comm);
    
    if (*global_n_p <= 0) {
        MPI_Finalize();
        exit(-1);
    }
    
    *local_n_p = *global_n_p/p;
    
}

void Crear_lista(int local_A[], int local_n, int my_rank) {
    int i;
    
    srandom(my_rank+1);
    for (i = 0; i < local_n; i++)
        local_A[i] = random() % RMAX;
    
}

void Imprimir_lista(int local_A[], int local_n, int my_rank, int p, MPI_Comm comm) {
    int* A = NULL;
    int i, n;
    
    if (my_rank == 0) {
        n = p*local_n;
        A = (int*) malloc(n*sizeof(int));
        MPI_Gather(local_A, local_n, MPI_INT, A, local_n, MPI_INT, 0, comm);
        printf("Lista:\n");
        for (i = 0; i < n; i++)
            printf("%d ", A[i]);
        printf("\n\n");
        free(A);
    } else {
        MPI_Gather(local_A, local_n, MPI_INT, A, local_n, MPI_INT, 0, comm);
    }
    
}

int Comparar(const void* a_p, const void* b_p) {
    int a = *((int*)a_p);
    int b = *((int*)b_p);
    
    if (a < b)
        return -1;
    else if (a == b)
        return 0;
    else
        return 1;
}


void MergeSplit_low(int local_A[], int temp_B[], int temp_C[], int local_n) {
    
    int ai, bi, ci;
    
    ai = 0;
    bi = 0;
    ci = 0;
    while (ci < local_n) {
        if (local_A[ai] <= temp_B[bi]) {
            temp_C[ci] = local_A[ai];
            ci++; ai++;
        } else {
            temp_C[ci] = temp_B[bi];
            ci++; bi++;
        }
    }
    
    memcpy(local_A, temp_C, local_n*sizeof(int));
}

void MergeSplit_high(int local_A[], int temp_B[], int temp_C[], int local_n) {
    
    int ai, bi, ci;
    
    ai = local_n-1;
    bi = local_n-1;
    ci = local_n-1;
    while (ci >= 0) {
        if (local_A[ai] >= temp_B[bi]) {
            temp_C[ci] = local_A[ai];
            ci--; ai--;
        } else {
            temp_C[ci] = temp_B[bi];
            ci--; bi--;
        }
    }
    
    memcpy(local_A, temp_C, local_n*sizeof(int));
}

void Odd_even(int local_A[], int temp_B[], int temp_C[], int local_n, int phase,
              int even_partner, int odd_partner, int my_rank, int p, MPI_Comm comm) {
    
    MPI_Status status;
    
    if (phase % 2 == 0) {
        if (even_partner >= 0) {
            MPI_Sendrecv(local_A, local_n, MPI_INT, even_partner, 0, temp_B, local_n,
                         MPI_INT, even_partner, 0, comm, &status);
            if (my_rank % 2 != 0)
                MergeSplit_high(local_A, temp_B, temp_C, local_n);
            else
                MergeSplit_low(local_A, temp_B, temp_C, local_n);
        }
    } else {
        if (odd_partner >= 0) {
            MPI_Sendrecv(local_A, local_n, MPI_INT, odd_partner, 0, temp_B, local_n,
                         MPI_INT, odd_partner, 0, comm, &status);
            if (my_rank % 2 != 0)
                MergeSplit_low(local_A, temp_B, temp_C, local_n);
            else
                MergeSplit_high(local_A, temp_B, temp_C, local_n);
        }
    }
}

void Odd_Even_Sort(int local_A[], int local_n, int my_rank,
          int p, MPI_Comm comm) {
    int phase;
    int *temp_B, *temp_C;
    int even_partner;
    int odd_partner;
    
    temp_B = (int*) malloc(local_n*sizeof(int));
    temp_C = (int*) malloc(local_n*sizeof(int));
    
    if (my_rank % 2 != 0) {
        even_partner = my_rank - 1;
        odd_partner = my_rank + 1;
        if (odd_partner == p) odd_partner = -1;
    } else {
        even_partner = my_rank + 1;
        if (even_partner == p) even_partner = -1;
        odd_partner = my_rank-1;
    }
    
    qsort(local_A, local_n, sizeof(int), Comparar);
    
    for (phase = 0; phase < p; phase++)
        Odd_even(local_A, temp_B, temp_C, local_n, phase, even_partner, odd_partner, my_rank, p, comm);
    
    free(temp_B);
    free(temp_C);
}


int main(int argc, char* argv[]) {
    int my_rank, p;
    int *local_A;
    int global_n;
    int local_n;
    MPI_Comm comm;
    double start, finish;
    
    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);
    
    Get_input(argc, argv, &global_n, &local_n, my_rank, p, comm);
    local_A = (int*) malloc(local_n*sizeof(int));

    Crear_lista(local_A, local_n, my_rank);
    //Imprimir_lista(local_A, local_n, my_rank, p, comm);

    
    start = MPI_Wtime();
    Odd_Even_Sort(local_A, local_n, my_rank, p, comm);
    finish = MPI_Wtime();
    if (my_rank == 0){
        printf("Tiempo: %e \n", finish-start);
    }
    //Imprimir_lista(local_A, local_n, my_rank, p, comm);
    
    free(local_A);
    
    MPI_Finalize();
    
    return 0;
}
