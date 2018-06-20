#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <time.h>

#include <stdio.h>
#include <stdlib.h>

#define N 5
#define BLOCK_DIM 10

using namespace std;
__global__ 
void sum_Matrices_Normal (int *a, int *b, int *c) {
    int columna = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    int id = columna + fila * N;
    if (columna < N && fila < N) {
        c[id] = a[id] + b[id];
    }

}

__global__ 
void sum_Matrices_fila (int *a, int *b, int *c) {
    int columna = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    for(int i=columna; i<N; i++){
        int id = i + fila * N;
        c[id] = a[id] + b[id];
    }
}

__global__ 
void sum_Matrices_columna (int *a, int *b, int *c) {
    int columna = blockIdx.x * blockDim.x + threadIdx.x;
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    for(int i=fila; i<N; i++){
        int id = columna + i * N;
        c[id] = a[id] + b[id];
    }
}

void imprimir_Matriz(int matrix[N][N]){
    for(int i=0;i<N;i++){
        for(int j=0; j<N; j++){
            cout<<matrix[i][j]<<' ';
        }
        cout<<endl;
    }
}

void imprimir_vector(int vector[N]){
    for(int j=0; j<N; j++){
        cout<<vector[j]<<' ';
    }
}



int main() {
    int a[N][N], b[N][N], c[N][N];
    int *dev_a, *dev_b, *dev_c;

    int size = N * N * sizeof(int);
    srand(time(NULL));
    for(int i=0; i<N; i++)
        for (int j=0; j<N; j++){
            a[i][j] = 1;
            b[i][j] = 1;
        }

    imprimir_Matriz(a);
    cout<<endl;
    imprimir_Matriz(b);


    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(N,N); // cuantos threads se ejecutaran juntos y que compartiran memoria en un sigle proccessor
    dim3 dimGrid(1,1); // un grupo de thread block que se ejecutan en un sigle cuda program logically in parallel
    sum_Matrices_Normal<<<dimGrid,dimBlock>>>(dev_a,dev_b,dev_c);
    //sum_Matrices_fila<<<dimGrid,dimBlock>>>(dev_a,dev_b,dev_c);
    //sum_Matrices_columna<<<dimGrid,dimBlock>>>(dev_a,dev_b,dev_c);

    cudaDeviceSynchronize();
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    cout<<endl;
    for(int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            printf("%d ", c[i][j] );
        }
        printf("\n");
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
