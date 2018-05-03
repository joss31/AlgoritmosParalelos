#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "time.h"

void Get_input(int argc, char* argv[], int* thread_count_p, int* m_p, int* n_p)  {
    *thread_count_p = strtol(argv[1], NULL, 10);
    *m_p = strtol(argv[2], NULL, 10);
    *n_p = strtol(argv[3], NULL, 10);
    
}

void Crear_matrix(double A[], int m, int n) {
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            A[i*n+j] = random()/((double) RAND_MAX);
}

void Crear_vector(double x[], int n) {
     int i;
     for (i = 0; i < n; i++)
        x[i] = random()/((double) RAND_MAX);
}

void Imprimir_matrix( char* title, double A[], int m, int n) {
    int    i, j;
    
    printf("%s\n", title);
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++)
            printf("%4.1f ", A[i*n + j]);
        printf("\n");
     }
}

void Imprimir_vector(char* title, double y[], double m) {
    int    i;
    
    printf("%s\n", title);
    for (i = 0; i < m; i++)
        printf("%4.1f ", y[i]);
    printf("\n");
}

void Mult_mat_vec(double A[], double x[], double y[], int m, int n, int thread_count) {
    int i, j, temp;
    
#  pragma omp parallel for num_threads(thread_count)  \
     default(none) private(i, j, temp)  shared(A, x, y, m, n)
     for (i = 0; i < m; i++) {
          y[i] = 0.0;
          for (j = 0; j < n; j++) {
                temp = A[i*n+j]*x[j];
                y[i] += temp;
          }
     }
    
}

int main(int argc, char* argv[]) {
    int      thread_count;
    int      m, n;
    double* A;
    double* x;
    double* y;
    double start, finish;
    
    Get_input(argc, argv, &thread_count, &m, &n);

    A = malloc(m*n*sizeof(double));
    x = malloc(n*sizeof(double));
    y = malloc(m*sizeof(double));
    
    Crear_matrix(A, m, n);
    //Imprimir_matrix("Matriz:", A, m, n);
    Crear_vector(x, n);
    //Imprimir_vector("Vector:", x, n);
    
    start = omp_get_wtime();
    Mult_mat_vec(A, x, y, m, n, thread_count);
    finish = omp_get_wtime();
    
    printf("tiempo = %e seconds\n", finish - start);
    
    //Imprimir_vector("Resultado: ", y, m);
    

    free(A);
    free(x);
    free(y);

    return 0;
}
