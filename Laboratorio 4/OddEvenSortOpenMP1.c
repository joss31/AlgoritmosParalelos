#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "time.h"

const int RMAX = 100;
int thread_count;

void Crear_list(int a[], int n) {
    int i;
    
    srandom(1);
    for (i = 0; i < n; i++)
        a[i] = random() % RMAX;
}

void Imprimir_list(int a[], int n, char* title) {
    int i;
    
    printf("%s:\n", title);
    for (i = 0; i < n; i++)
        printf("%d ", a[i]);
    printf("\n\n");
}

void Odd_evenOpenMP(int a[], int n) {
    int phase, i, tmp;
    
    for (phase = 0; phase < n; phase++) {
        if (phase % 2 == 0)
#           pragma omp parallel for num_threads(thread_count) \
                default(none) shared(a, n) private(i, tmp)
            for (i = 1; i < n; i += 2) {
                if (a[i-1] > a[i]) {
                    tmp = a[i-1];
                    a[i-1] = a[i];
                    a[i] = tmp;
                }
            }
        else
#           pragma omp parallel for num_threads(thread_count) \
                default(none) shared(a, n) private(i, tmp)
            for (i = 1; i < n-1; i += 2) {
                if (a[i] > a[i+1]) {
                    tmp = a[i+1];
                    a[i+1] = a[i];
                    a[i] = tmp;
                }
            }
    }
}

int main(int argc, char* argv[]) {
   int  n;
   int* a;
   double start, finish;
   
   thread_count = strtol(argv[1], NULL, 10);
   n = strtol(argv[2], NULL, 10);

   a = malloc(n*sizeof(int));
   Crear_list(a, n);
   //Imprimir_list(a, n, "Lista");

   start = omp_get_wtime();
   Odd_evenOpenMP(a, n);
   finish = omp_get_wtime();

   //Imprimir_list(a, n, "Resultado: ");
   printf("tiempo = %e seconds\n", finish - start);

   free(a);
   return 0;
}
