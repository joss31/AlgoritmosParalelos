#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

const int RMAX = 100;
int thread_count;

void Generate_matrix(double A[], int m, int n);
void Generate_vector(double x[], int n);
void Print_matrix(char* title, double A[], int m, int n);
void Print_vector(char* title, double y[], double m);
void Mat_vectOpenMP(double A[], double x[], double y[], int m, int n);

/*------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
   int     m, n;
   double* A;
   double* x;
   double* y;

   thread_count = strtol(argv[1], NULL, 10);
   m = strtol(argv[2], NULL, 10);
   n = strtol(argv[3], NULL, 10);

   A = malloc(m*n*sizeof(double));
   x = malloc(n*sizeof(double));
   y = malloc(m*sizeof(double));
   
   Generate_matrix(A, m, n);
   Print_matrix("generated", A, m, n); 
   Generate_vector(x, n);
   Print_vector("generated", x, n); 
 

   Mat_vectOpenMP(A, x, y, m, n);

   Print_vector("Results", y, m);

   free(A);
   free(x);
   free(y);

   return 0;
}  /* main */

void Generate_matrix(double A[], int m, int n) {
   int i, j;
   for (i = 0; i < m; i++)
      for (j = 0; j < n; j++)
         A[i*n+j] = random() % RMAX;
}  

void Generate_vector(double x[], int n) {
   int i;
   for (i = 0; i < n; i++)
      x[i] = random() % RMAX;
} 

void Mat_vectOpenMP(double A[], double x[], double y[], int m, int n) {
   int i, j;

#  pragma omp parallel for num_threads(thread_count)  \
      default(none) private(i, j)  shared(A, x, y, m, n)
   for (i = 0; i < m; i++) {
      y[i] = 0.0;
      for (j = 0; j < n; j++)
         y[i] += A[i*n+j]*x[j];
   }
}

void Print_matrix( char* title, double A[], int m, int n) {
   int   i, j;

   printf("%s\n", title);
   for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++)
         printf("%4.1f ", A[i*n + j]);
      printf("\n");
   }
}

void Print_vector(char* title, double y[], double m) {
   int   i;

   printf("%s\n", title);
   for (i = 0; i < m; i++)
      printf("%4.1f ", y[i]);
   printf("\n");
}