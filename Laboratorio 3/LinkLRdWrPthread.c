#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

int thread_count;
struct list_node_s** head_pp;
pthread_mutex_t mutex;
pthread_mutex_t mutex2;
pthread_rwlock_t rwlock;
pthread_mutex_t head_p_mutex;

long operaciones=100000;
double member=99.9;
double insert=0.05;
double del=0.05;
struct timeval t_ini, t_fin;
double secs;

struct list_node_s
{
    int data;
    struct list_node_s* next;
    pthread_mutex_t mutex;
};

double timeval_diff(struct timeval *a, struct timeval *b){
    
    return (double)(a->tv_sec + (double)a->tv_usec/100) - (double)(b->tv_sec + (double)b->tv_usec/100);
}

int Member(int value){
    struct list_node_s* curr_p=*head_pp;
    
    while(curr_p!=NULL && curr_p->data < value)
        curr_p=curr_p->next;
    if(curr_p == NULL || curr_p->data >value)
        return 0;
    else
        return 1;
}

int Insert(int value){
    struct list_node_s* curr_p= *head_pp;
    struct list_node_s* pred_p= NULL;
    struct list_node_s* temp_p;
    while(curr_p != NULL && curr_p->data<value){
        pred_p=curr_p;
        curr_p=curr_p->next;
    }
    if(curr_p == NULL || curr_p->data > value){
        temp_p=malloc(sizeof(struct list_node_s));
        temp_p->data=value;
        temp_p->next=curr_p;
        if (pred_p == NULL)
            *head_pp=temp_p;
        else
            pred_p->next=temp_p;
        return 1;
    }
    else
        return 0;
}

int Delete(int value){
    struct list_node_s* curr_p=*head_pp;
    struct list_node_s* pred_p= NULL;
    while(curr_p != NULL && curr_p->data < value){
        pred_p=curr_p;
        curr_p=curr_p->next;
    }
    if(curr_p != NULL && curr_p->data == value){
        if(pred_p == NULL)
        {
            *head_pp=curr_p->next;
            free(curr_p);
        }
        else
        {
            pred_p->next=curr_p->next;
            free(curr_p);
        }
        return 1;
    }
    else
        return 0;
}

void* RW(void* r){
    long ops=(long) r;
    
    for(int j=0;j<ops*member/100;j++){
        pthread_rwlock_rdlock(&rwlock);
        Member(rand()%10000);
        pthread_rwlock_unlock(&rwlock);
    }
    
    for(int j=0;j<ops*insert/100;j++){
        pthread_rwlock_wrlock(&rwlock);
        Insert(rand()%10000);
        pthread_rwlock_unlock(&rwlock);
    }
    
    for(int j=0;j<ops*del/100;j++){
        pthread_rwlock_wrlock(&rwlock);
        Delete(rand()%10000);
        pthread_rwlock_unlock(&rwlock);
    }
}

int main(int argc,char* argv[]){
    long thread;
    pthread_t* thread_handles;
    struct list_node_s* head;

    head=malloc(sizeof(struct list_node_s));
    head->data=0;
    head->next=NULL;
    head_pp=&head;
    
    thread_count=strtol(argv[1],NULL,10);
    thread_handles=(pthread_t*) malloc (thread_count*sizeof(pthread_t));
    
    for(int i =0; i<1000;++i){
        Insert(i);
    }
     ///Read-Write
     gettimeofday(&t_ini, NULL);
     for(thread=0;thread<thread_count;thread++){
         pthread_create(&thread_handles[thread],NULL,RW,(void *)thread);
     }
     for(thread=0;thread<thread_count;thread++){
        pthread_join(thread_handles[thread],NULL);
     }
     gettimeofday(&t_fin, NULL);
     secs = timeval_diff(&t_fin, &t_ini);
     printf("%.16g Read-Write segundos\n", secs );
     free(thread_handles);
     
    
    return 0;
}
