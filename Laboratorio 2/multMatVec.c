#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

int my_rank, comm_sz;
MPI_Comm comm;


void Get_input(int argc, char* argv[], int* m_p, int* local_m_p,
              int* n_p, int* local_n_p){

    int local_ok = 1;
    
    if (my_rank == 0) {
        if (argc != 2)
        *m_p = *n_p = 0;
        else
        *m_p = *n_p = strtol(argv[1], NULL, 10);
    }
    
    MPI_Bcast(m_p, 1, MPI_INT, 0, comm);
    MPI_Bcast(n_p, 1, MPI_INT, 0, comm);
    
    if (*m_p <= 0 || *n_p <= 0 || *m_p % comm_sz != 0 || *n_p % comm_sz != 0)
        local_ok = 0;
    
    *local_m_p = *m_p/comm_sz;
    *local_n_p = *n_p/comm_sz;
}

void AllocateArrays(double** local_A_pp, double** local_x_pp,
                     double** local_y_pp, int local_m, int n, int local_n){
  int local_ok = 1;
    
    *local_A_pp = malloc(local_m*n*sizeof(double));
    *local_x_pp = malloc(local_n*sizeof(double));
    *local_y_pp = malloc(local_m*sizeof(double));
    
    if (*local_A_pp == NULL || local_x_pp == NULL || local_y_pp == NULL)
        local_ok = 0;
   
}

void Crear_matriz(double local_A[], int local_m, int n){
    int i, j;
    
    for (i = 0; i < local_m; i++)
        for (j = 0; j < n; j++)
            local_A[i*n + j] = ((double) random())/((double) RAND_MAX);

    for (i = 0; i < local_m; i++)
        for (j = 0; j < n; j++)
            local_A[i*n + j] = my_rank + i;
}


void Crear_vector(double local_x[], int local_n){
    int i;

    for (i = 0; i < local_n; i++)
        local_x[i] = ((double) random())/((double) RAND_MAX);

    for (i = 0; i < local_n; i++)
        local_x[i] = my_rank + 1;
    

}


void Imprimir_matriz(char title[], double local_A[], int m, int local_m, int n){
    double* A = NULL;
    int i, j, local_ok = 1;
    
    if (my_rank == 0) {
        A = malloc(m*n*sizeof(double));
        if (A == NULL) local_ok = 0;
      
        MPI_Gather(local_A, local_m*n, MPI_DOUBLE, A, local_m*n, MPI_DOUBLE, 0, comm);
        printf("\nThe matrix %s\n", title);
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++)
            printf("%f ", A[i*n+j]);
            printf("\n");
        }
        printf("\n");
        free(A);
        
    } else {
        
        MPI_Gather(local_A, local_m*n, MPI_DOUBLE, A, local_m*n, MPI_DOUBLE, 0, comm);
    }
}


void Imprimir_vector(char title[], double local_vec[], int n, int local_n){
    double* vec = NULL;
    int i, local_ok = 1;
    
    if (my_rank == 0) {
        vec = malloc(n*sizeof(double));
        if (vec == NULL) local_ok = 0;
  
        MPI_Gather(local_vec, local_n, MPI_DOUBLE, vec, local_n, MPI_DOUBLE, 0, comm);
        printf("\nThe vector %s\n", title);
        for (i = 0; i < n; i++)
        printf("%f ", vec[i]);
        printf("\n");
        free(vec);
        
    }  else {
        
        MPI_Gather(local_vec, local_n, MPI_DOUBLE, vec, local_n, MPI_DOUBLE, 0, comm);
    }
    
}

void Multiplicar_mat_vec(double local_A[], double local_x[],double local_y[],
                   double x[], int m, int local_m, int n, int local_n){
    
    int local_i, j;
    
    MPI_Allgather(local_x, local_n, MPI_DOUBLE, x, local_n, MPI_DOUBLE, comm);
    
    for (local_i = 0; local_i < local_m; local_i++) {
        local_y[local_i] = 0.0;
        for (j = 0; j < n; j++)
            local_y[local_i] += local_A[local_i*n+j]*x[j];
    }
}

int main(int argc, char* argv[]) {
    double* local_A;
    double* local_x;
    double* local_y;
    double* x;
    int m, local_m, n, local_n;
    double start, finish, total_time, time;
    
    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);
    
    Get_input(argc, argv, &m, &local_m, &n, &local_n);
    AllocateArrays(&local_A, &local_x, &local_y, local_m, n, local_n);
    
    srandom(my_rank);
    
    //crear matriz
    Crear_matriz(local_A, local_m, n);
    //Imprimir_matriz("A", local_A, m, local_m, n);

    //crear vector
    Crear_vector(local_x, local_n);
    //Imprimir_vector("x", local_x, n, local_n);

    x = malloc(n*sizeof(double));
    MPI_Barrier(comm);
    start = MPI_Wtime();
    
    //multiplicación de matriz vector
    Multiplicar_mat_vec(local_A, local_x, local_y, x, m, local_m, n, local_n);
    finish = MPI_Wtime();
    total_time = finish-start;
    MPI_Reduce(&total_time, &time, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    
    
    //Imprimir_vector("y", local_y, m, local_m);
    
    if (my_rank == 0) {
        printf("%e\n", time);

    }
  
    free(local_A);
    free(local_x);
    free(local_y);
    free(x);
    MPI_Finalize();
    return 0;
}
