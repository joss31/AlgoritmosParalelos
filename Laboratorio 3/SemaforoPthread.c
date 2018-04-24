#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include "time.h"

#define BARRIER_COUNT 100

struct timeval t_ini, t_fin;
double total_time;

int thread_count;
int counter;
sem_t barrier_sems[BARRIER_COUNT];
sem_t count_sem;

double timeval_diff(struct timeval *a, struct timeval *b)
{
     return (double)(a->tv_sec + (double)a->tv_usec/100000) - (double)(b->tv_sec + (double)b->tv_usec/100000);
}

void *Thread_work(void* rank) {

     for (int i = 0; i < BARRIER_COUNT; i++) {
          sem_wait(&count_sem);
          if (counter == thread_count - 1) {
              counter = 0;
              sem_post(&count_sem);
              for (int j = 0; j < thread_count-1; j++){
                    sem_post(&barrier_sems[i]);
              }
              
          } else {
                counter++;
                sem_post(&count_sem);
                sem_wait(&barrier_sems[i]);
          }

     }
     
     return NULL;
}

int main(int argc, char* argv[]) {
    long         thread, i;
    pthread_t* thread_handles; 
    
    thread_count = strtol(argv[1], NULL, 10);

    thread_handles = malloc (thread_count*sizeof(pthread_t));
    
    for (i = 0; i < BARRIER_COUNT; i++)
        sem_init(&barrier_sems[i], 0, 0);
    sem_init(&count_sem, 0, 1);

    gettimeofday(&t_ini, NULL);

    for (thread = 0; thread < thread_count; thread++){
        pthread_create(&thread_handles[thread], (pthread_attr_t*) NULL, Thread_work, (void*) thread);
    }
    for (thread = 0; thread < thread_count; thread++) {
        pthread_join(thread_handles[thread], NULL);
    }
    
    gettimeofday(&t_fin, NULL);
    
    total_time = timeval_diff(&t_fin, &t_ini);
    printf("Tiempo = %.16g \n", total_time * 1000.0/thread_count);

    sem_destroy(&count_sem);
    for (i = 0; i < BARRIER_COUNT; i++)
        sem_destroy(&barrier_sems[i]);
    free(thread_handles);
    
    return 0;
}
