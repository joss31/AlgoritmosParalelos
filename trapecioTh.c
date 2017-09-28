#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

int     thread_count;
double  a, b, h;
int     n, local_n;

pthread_mutex_t   mutex;
double  total;

void *Thread_work(void* rank);
double Trap(double local_a, double local_b, int local_n,
           double h); 
double f(double x); 

int main(int argc, char** argv) {
    long        i;
    pthread_t*  thread_handles;  

    total = 0.0;
    if (argc != 2) {
       fprintf(stderr, "usage: %s <numero de threads>\n", argv[0]);
       exit(0);
    }
    thread_count = strtol(argv[1], NULL, 10);
    printf("Ingresa a, b, n\n");
    scanf("%lf %lf %d", &a, &b, &n);
    h = (b-a)/n;
    local_n = n/thread_count;

    thread_handles = malloc (thread_count*sizeof(pthread_t));

    pthread_mutex_init(&mutex, NULL);

    for (i = 0; i < thread_count; i++) {
         pthread_create(&thread_handles[i], NULL, Thread_work, 
               (void*) i);
    }

    for (i = 0; i < thread_count; i++) {
        pthread_join(thread_handles[i], NULL);
    }

    printf("Con n = %d trapezoides, la estimacion\n",
         n);
    printf("de la integral desde %f y %f = %19.15e\n",
         a, b, total);

    pthread_mutex_destroy(&mutex);
    free(thread_handles);

    return 0;
}


void *Thread_work(void* rank) {
    double      local_a;  
    double      local_b;  
    double      my_int;
    long        my_rank = (long) rank;

    local_a = a + my_rank*local_n*h;
    local_b = local_a + local_n*h;

    my_int = Trap(local_a, local_b, local_n, h);

    pthread_mutex_lock(&mutex);
    total += my_int;
    pthread_mutex_unlock(&mutex);

    return NULL;

} 

double Trap(
          double  local_a   ,
          double  local_b   ,
          int     local_n   ,
          double  h         ) {

    double integral; 
    double x;
    int i;

    integral = (f(local_a) + f(local_b))/2.0;
    x = local_a;
    for (i = 1; i <= local_n-1; i++) {
        x = local_a + i*h;
        integral += f(x);
    }
    integral = integral*h;
    return integral;
} 


double f(double x) {
    double return_val;

    return_val = x*x;
    return return_val;
}
