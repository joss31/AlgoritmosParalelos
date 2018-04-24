#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "time.h"

long thread_count;
long long n;
double sum;
pthread_mutex_t mutex;
struct timeval t_ini, t_fin;
double total_time;

double timeval_diff(struct timeval *a, struct timeval *b)
{
    return (double)(a->tv_sec + (double)a->tv_usec/100000) - (double)(b->tv_sec + (double)b->tv_usec/100000);
}

void Get_input(int argc, char* argv[]) {
    thread_count = strtol(argv[1], NULL, 10);
    n = strtoll(argv[2], NULL, 10);
}

void* Thread_sum(void* rank) {
    long my_rank = (long) rank;
    double factor;
    long long i;
    long long my_n = n/thread_count;
    long long my_first_i = my_n*my_rank;
    long long my_last_i = my_first_i + my_n;
    double my_sum = 0.0;
    
    if (my_first_i % 2 == 0)
        factor = 1.0;
    else
        factor = -1.0;
    
    for (i = my_first_i; i < my_last_i; i++, factor = -factor) {
        my_sum += factor/(2*i+1);
    }
    pthread_mutex_lock(&mutex);
    sum += my_sum;
    pthread_mutex_unlock(&mutex);
    
    return NULL;
}


double Serial_pi(long long n) {
    double sum = 0.0;
    long long i;
    double factor = 1.0;
    
    for (i = 0; i < n; i++, factor = -factor) {
        sum += factor/(2*i+1);
    }
    return 4.0*sum;
    
}

int main(int argc, char* argv[]) {
    long thread;
    pthread_t* thread_handles;

    Get_input(argc, argv);

    thread_handles = (pthread_t*) malloc (thread_count*sizeof(pthread_t)); 
    pthread_mutex_init(&mutex, NULL);
    sum = 0.0;

    gettimeofday(&t_ini, NULL);
     
    for (thread = 0; thread < thread_count; thread++){
        pthread_create(&thread_handles[thread], NULL, Thread_sum, (void*)thread);
    }
    for (thread = 0; thread < thread_count; thread++){
        pthread_join(thread_handles[thread], NULL);
    }

    gettimeofday(&t_fin, NULL);
    total_time = timeval_diff(&t_fin, &t_ini);
    
    sum = 4.0*sum;
    printf("Con n = %lld ,\n", n);
    printf("          Estimación de Pi = %.15f\n", sum);
    printf("                    Tiempo = %.16g \n", total_time * 1000.0/thread_count);

    gettimeofday(&t_ini, NULL);
    
    sum = Serial_pi(n);
    
    gettimeofday(&t_fin, NULL);
    total_time = timeval_diff(&t_fin, &t_ini);
    
    printf("  Estimacioón de PiSerial = %.15f\n", sum);
    printf("                 Tiempo   = %.16g \n", total_time * 1000.0/thread_count);
    printf("                       pi = %.15f\n", 4.0*atan(1.0));
    

    pthread_mutex_destroy(&mutex);
    free(thread_handles);
    
    return 0;
}
