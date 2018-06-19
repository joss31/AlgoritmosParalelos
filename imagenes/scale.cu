#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stdlib.h>
#include <iostream>
#include "lodepng.h"
using namespace std;

__global__
void PictureKernell(unsigned char* d_Pin, unsigned char* d_Pout, int n, int m){
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int new_pos;
  if((y < n) && (x < m)) {
    new_pos = (y*m+x)*4;
    d_Pout[new_pos] = 2*d_Pin[new_pos];
    d_Pout[new_pos+1] = 2*d_Pin[new_pos+1];
    d_Pout[new_pos+2] = 2*d_Pin[new_pos+2];
    d_Pout[new_pos+3] = d_Pin[new_pos+3];
  }
}

__global__
void PictureKernelll(unsigned char* d_Pin, unsigned char* d_Pout, int n, int m){
	/*int Row = blockIdx.y*blockDim.y + threadIdx.y;
	int Col = blockIdx.x*blockDim.x + threadIdx.x;

	if ((Row < height) && (Col < width)) {
		d_Pout[Row*width+Col] = 2.0*d_Pin[Row*width+Col];
	}
  */
  
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int new_pos;
  if((y < n) && (x < m)) {
    new_pos = (y*m+x);
    d_Pout[new_pos] = 2*d_Pin[new_pos];
  }
}


__global__
void PictureKernel1D(unsigned char* d_Pin, unsigned char* d_Pout, int n, int m){
  int Row = blockIdx.x * blockDim.x + threadIdx.x;
  if(Row < n*m*4) {
    d_Pout[Row] = 2*d_Pin[Row];
  }
}

void Picture(unsigned char* Pin, unsigned char* Pout, int n, int m){
  unsigned char* d_Pout, *d_Pin;
  long int size = n*m*4;
  cudaMalloc((void **) &d_Pin,size);
  cudaMemcpy(d_Pin, Pin, size, cudaMemcpyHostToDevice);
  cudaMalloc((void **) &d_Pout,size);

   dim3 gridDim((m-1)/8+1,(n-1)/16+1,1);
   dim3 blockDim(8,16,1);
   //dim3 gridDim2((m-1)/8+1,(n-1)/16+1,1);
   //dim3 blockDim2(8,16,1);
   PictureKernell<<<gridDim,blockDim>>>(d_Pin,d_Pout,n,m);
   //PictureKernelll<<< gridDim2 , blockDim2 >>>(d_Pin,d_Pout,n,m);
  //PictureKernel1D<<<(size-1)/256+1,256>>>(d_Pin,d_Pout,n,m);

  cudaMemcpy(Pout, d_Pout, size, cudaMemcpyDeviceToHost);
  cudaFree(d_Pin); cudaFree(d_Pout);
}

int main(int argc, char * argv[] ){
  unsigned char *image, *out_image;
  int i;
  char name_in[100], name_out[100];
  unsigned width, height;
  if(argv[1] == NULL or argv[2] == NULL)
    cout << "Usage\n inverse.cu [input image] [output image]\n";
  strcpy(name_in,argv[1]);
  strcpy(name_out,argv[2]);
  i = lodepng_decode32_file(&image, &width, &height, name_in);
  if(i < 0) printf("NO\n");
  out_image = (unsigned char*) malloc(width*height*4);
  Picture(image,out_image,height,width);
  lodepng_encode32_file(name_out,out_image,width,height);

  free(image);
  free(out_image);
  return 0;
}
