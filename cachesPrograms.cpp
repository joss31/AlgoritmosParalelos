#include <time.h>
#include <random>

#define MAX 509

int main(void) {
    
    double A[MAX][MAX], x[MAX], y[MAX];
    
    for(int i = 0; i < MAX; i++){
        for(int j = 0; j < MAX; j++){
            A[i][j] = rand() % 10;
        }
        x[i] = rand() % 10;
        y[i] = 0;
    }
    
    clock_t tStart = clock();
    
    for(int i = 0; i < MAX; i++){
        for(int j = 0; j < MAX; j++){
            y[i] += A[i][j] + x[j];
        }
    }
    
    printf("Time taken: %.8fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
    
    tStart = clock();
    for(int j = 0; j < MAX; j++){
        for(int i=0; i<MAX; i++){
            y[i] += A[i][j] + x[j];
        }
    }
    
    printf("Time taken: %.8fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
    return 0;
}