#include <stdio.h>
#include <iostream>
#define N 5

using namespace std;
__global__
void matrix_vector_mult(float *A, float *B, float *C, int n){
  int i = threadIdx.x + blockDim.x * blockIdx.x, j;
  if(i < n){
    C[i] = 0;
    for(j = 0; j < n; j++){
       C[i] += A[i*n+j] * B[j];
    }
  }
  
}


int main(){
    int i;  //int j;
    float *A,*B,*C;
    A = (float*) malloc(N*N*sizeof(float));
    B = (float*) malloc(N*sizeof(float));
    C = (float*) malloc(N*sizeof(float));
    for(i = 0; i < N*N; i++){
    //for(j = 0; j < N; j++)
    //h_A[i*N+j] = 1;
        A[i]=1;
    }
    for(i = 0; i < N; i++){
        B[i] = 1;
        C[i] = 0;
    }

    int size = N*sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **) &d_A, size*N);
    cudaMemcpy(d_A,A,size*N,cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_B, size);
    cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);
    cudaMalloc((void **) &d_C, size);
    matrix_vector_mult<<<ceil(N/256.0), 256>>>(d_A,d_B,d_C,N);

    cudaMemcpy(C,d_C,size,cudaMemcpyDeviceToHost);

    for(i = 0; i < N; i++){
        cout<<C[i]<<" ";
    }
    cout<<endl;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
