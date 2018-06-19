#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <iomanip>

uchar4        	*device_RGBA_Imagen__;
unsigned char 	*device_GRIS_Imagen__;

using namespace std;
using namespace cv;
Mat imagenRGBA;
Mat imagenGris;

size_t numFilas() { 
	return imagenRGBA.rows; }
size_t numCols() { 
	return imagenRGBA.cols; 	}	

void cargar_Imagen_host_To_Device(uchar4 **inputImage, unsigned char **greyImage, uchar4 **device_RGBA_Imagen, unsigned char **device_GRIS_Imagen, const std::string &filename) {
	Mat imagen 		= imread(filename.c_str(), CV_LOAD_IMAGE_COLOR); 
	cvtColor(imagen, imagenRGBA, CV_BGR2RGBA);
	imagenGris.create(imagen.rows, imagen.cols, CV_8UC1);	//asignar memoria para la salida acorde a canales
	
	*inputImage = (uchar4 *)imagenRGBA.ptr<unsigned char>(0);		
	*greyImage  = imagenGris.ptr<unsigned char>(0);
	
	const size_t numPixels = numFilas() * numCols();
	cudaMalloc(device_RGBA_Imagen, sizeof(uchar4) * numPixels);	// asignar memoria en el dispositivo para entrada y salida
	cudaMalloc(device_GRIS_Imagen, sizeof(unsigned char) * numPixels);
	cudaMemset(*device_GRIS_Imagen, 0, numPixels * sizeof(unsigned char)); 	//asegúrate de que no quede memoria tirada
	cudaMemcpy(*device_RGBA_Imagen, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);
	device_RGBA_Imagen__ = *device_RGBA_Imagen;				
	device_GRIS_Imagen__ = *device_GRIS_Imagen;
}

__global__
void rgba_a_gris_cuda_kernel(const uchar4* const rgbaImage, unsigned char* const greyImage, const int numFilas, const int numCols){
	int id				=	threadIdx.x + blockDim.x*blockIdx.x;
  	unsigned char r		=	rgbaImage[id].x;
  	unsigned char g		=	rgbaImage[id].y;
  	unsigned char b		=	rgbaImage[id].z;
  	greyImage[id]		=	0.21f * r +	0.71f * g +	0.07f * b;
}

int main(int argc, char **argv) {
	uchar4          *host_RGBA_Imagen, *device_RGBA_Imagen;          // x=R y=G; z=B; w=A    ==>  uchar4
	unsigned char   *host_GRIS_Imagen, *device_GRIS_Imagen;
	
	cargar_Imagen_host_To_Device(&host_RGBA_Imagen, &host_GRIS_Imagen, &device_RGBA_Imagen, &device_GRIS_Imagen, "input.jpg");
	
	const int blockThreadSize 			= 512;
	const int numberOfBlocks 			= 1 + ((numFilas()*numCols() - 1) / blockThreadSize); 			// Numerado/Denominador redondeado up
	const dim3 blockSize(blockThreadSize, 1, 1);
	const dim3 gridSize(numberOfBlocks , 1, 1);
	
	rgba_a_gris_cuda_kernel<<<gridSize, blockSize>>>(device_RGBA_Imagen, device_GRIS_Imagen, numFilas(), numCols());
	cudaDeviceSynchronize();
	
	size_t numPixels 					= numFilas()*numCols();
	cudaMemcpy(host_GRIS_Imagen, device_GRIS_Imagen, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);
	Mat output(numFilas(), numCols(), CV_8UC1, (void*)host_GRIS_Imagen);
	imwrite("inputGris.jpg", output);
    return 0;
}
