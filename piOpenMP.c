#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h> 


int main(int argc, char* argv[]) {
   long long n, i;
   int thread_count;
   double factor;
   double sum = 0.0;

   thread_count = strtol(argv[1], NULL, 10);
   n = strtoll(argv[2], NULL, 10);

#  pragma omp parallel for num_threads(thread_count) \
      reduction(+: sum) private(factor)
   for (i = 0; i < n; i++) {
      factor = (i % 2 == 0) ? 1.0 : -1.0; 
      sum += factor/(2*i+1);

   }

   sum = 4.0*sum;
   printf("With n = %lld terms and %d threads,\n", n, thread_count);
   printf("   Our estimate of pi = %.14f\n", sum);

   return 0;
}  /* main */
