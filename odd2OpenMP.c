#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

const int RMAX = 100;


int thread_count;

void Generate_list(int a[], int n);
void Print_list(int a[], int n, char* title);
void Odd_even2OpenMP(int a[], int n);

int main(int argc, char* argv[]) {
   int  n;
   int* a;
   double start, finish;

   thread_count = strtol(argv[1], NULL, 10);
   n = strtol(argv[2], NULL, 10);

   a = malloc(n*sizeof(int));
   Generate_list(a, n);
  //Print_list(a, n, "Before sort");

   start = omp_get_wtime();
   Odd_even2OpenMP(a, n);
   finish = omp_get_wtime();

   //Print_list(a, n, "After sort");
   printf("time = %e seconds\n", finish - start);

   
   free(a);
   return 0;
} /* main */


void Generate_list(int a[], int n) {
   int i;

   srandom(1);
   for (i = 0; i < n; i++)
      a[i] = random() % RMAX;
} 

void Print_list(int a[], int n, char* title) {
   int i;

   printf("%s:\n", title);
   for (i = 0; i < n; i++)
      printf("%d ", a[i]);
   printf("\n\n");
}

void Odd_even2OpenMP(int a[], int n) {
   int phase, i, tmp;

#  pragma omp parallel num_threads(thread_count) \
      default(none) shared(a, n) private(i, tmp, phase)
   for (phase = 0; phase < n; phase++) {
      if (phase % 2 == 0)
# pragma omp for 
         for (i = 1; i < n; i += 2) {
            if (a[i-1] > a[i]) {
               tmp = a[i-1];
               a[i-1] = a[i];
               a[i] = tmp;
            }
         }
      else
# pragma omp for 
         for (i = 1; i < n-1; i += 2) {
            if (a[i] > a[i+1]) {
               tmp = a[i+1];
               a[i+1] = a[i];
               a[i] = tmp;
            }
         }
   }
}