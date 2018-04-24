#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "time.h"

int     thread_count;
int     m, n;
double* A;
double* x;
double* y;
struct timeval t_ini, t_fin;
double total_time;

double timeval_diff(struct timeval *a, struct timeval *b)
{
    return (double)(a->tv_sec + (double)a->tv_usec/10000000) - (double)(b->tv_sec + (double)b->tv_usec/10000000);
}

void Gen_matrix(double A[], int m, int n) {
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            A[i*n+j] = random()/((double) RAND_MAX);
}

void Gen_vector(double x[], int n) {
    int i;
    for (i = 0; i < n; i++)
        x[i] = random()/((double) RAND_MAX);
}

void Print_matrix( char* title, double A[], int m, int n) {
    int   i, j;
    
    printf("%s\n", title);
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++)
            printf("%6.3f ", A[i*n + j]);
        printf("\n");
    }
}

void Print_vector(char* title, double y[], double m) {
    int   i;
    
    printf("%s\n", title);
    for (i = 0; i < m; i++)
        printf("%6.3f ", y[i]);
    printf("\n");
}

void *Pth_mat_vect(void* rank) {
    long my_rank = (long) rank;
    int i;
    int j;
    int local_m = m/thread_count;
    int my_first_row = my_rank*local_m;
    int my_last_row = my_first_row + local_m;
    register int sub = my_first_row*n;
    double temp;
    
#  ifdef DEBUG
    printf("Thread %ld > local_m = %d, sub = %d\n",
           my_rank, local_m, sub);
#  endif
    
    for (i = my_first_row; i < my_last_row; i++) {
        y[i] = 0.0;
        for (j = 0; j < n; j++) {
            temp = A[sub++];
            temp *= x[j];
            y[i] += temp;
        }
    }
    
    return NULL;
}

int main(int argc, char* argv[]) {
   long       thread;
   pthread_t* thread_handles;

   thread_count = strtol(argv[1], NULL, 10);
   m = strtol(argv[2], NULL, 10);
   n = strtol(argv[3], NULL, 10);

#  ifdef DEBUG
   printf("thread_count =  %d, m = %d, n = %d\n", thread_count, m, n);
#  endif

   thread_handles = malloc(thread_count*sizeof(pthread_t));
   A = malloc(m*n*sizeof(double));
   x = malloc(n*sizeof(double));
   y = malloc(m*sizeof(double));
   
   Gen_matrix(A, m, n);

    gettimeofday(&t_ini, NULL);
    
   for (thread = 0; thread < thread_count; thread++)
      pthread_create(&thread_handles[thread], NULL, Pth_mat_vect, (void*) thread);

   for (thread = 0; thread < thread_count; thread++)
      pthread_join(thread_handles[thread], NULL);

    gettimeofday(&t_fin, NULL);
    total_time = timeval_diff(&t_fin, &t_ini);
    printf(" Tiempo = %.16g \n", total_time * 1000.0/thread_count);

   //Print_vector("The product is", y, m);

    free(A);
   free(x);
   free(y);

   return 0;
}
