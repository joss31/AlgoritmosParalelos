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

using namespace std;
using namespace cv;

uchar4        	*device_RGBA_Imagen__;
unsigned char 	*device_GRIS_Imagen__;

Mat imageInputRGBA;
Mat imageOutputRGBA;

uchar4 *d_inputImageRGBA__;
uchar4 *d_outputImageRGBA__;
        
float *h_filter__;

size_t numRows() { return imageInputRGBA.rows; }
size_t numCols() { return imageInputRGBA.cols; }


__global__
void blur_Kernel(const unsigned char* const in,	unsigned char* const out,	int numRows, int numCols, const float* const filter, const int filterWidth){
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(Col < numRows && Row< numRows){
		int pixVal =0;
		int pixels =0;
		for (int blurRow = -filterWidth; blurRow<filterWidth+1; ++blurRow){
			for(int blurCol = -filterWidth; blurCol< filterWidth+1; ++blurCol){
				int curRow= Row+blurRow;
				int curCol= Col+blurCol;
				if(curRow>-1 && curRow<numRows && curCol>-1 && curCol<numCols){
					pixVal+=in[curRow*numRows + curCol];
					pixels++;
				}
			}
		}
		out[Row*numRows+Col]=(unsigned char)(pixVal/pixels);
	}
    __syncthreads();

}

__global__
void separar_RGBA(const uchar4* const inputImageRGBA, int numRows, int numCols, unsigned char* const redChannel, unsigned char* const greenChannel, unsigned char* const blueChannel){
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = idx_y * numCols + idx_x;

    if (idx_x < numCols && idx_y < numRows){
        uchar4 input = inputImageRGBA[idx];
        __syncthreads();
        redChannel[idx] = input.x;
        greenChannel[idx] = input.y;
        blueChannel[idx] = input.z;
    }
}

__global__
void recombinar_RGBA(const unsigned char* const redChannel, const unsigned char* const greenChannel, const unsigned char* const blueChannel, uchar4* const outputImageRGBA, int numRows, int numCols) {
    const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
        return;

    unsigned char red   = redChannel[thread_1D_pos];
    unsigned char green = greenChannel[thread_1D_pos];
    unsigned char blue  = blueChannel[thread_1D_pos];

    uchar4 outputPixel = make_uchar4(red, green, blue, 255); // para el canal alpha, esta debe ser 255 sin transparente
    outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void asign_memoria_copiar_Device(const size_t numRowsImage, const size_t numColsImage, const float* const h_filter, const size_t filterWidth){

  cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage);
  cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage);
  cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage);
  cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth);

  cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice);
}

void filtro_a_realizar(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA, uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols, unsigned char *d_redBlurred, unsigned char *d_greenBlurred, unsigned char *d_blueBlurred, const int filterWidth){
    const dim3 blockSize(32, 32);
    const dim3 gridSize( (numCols + blockSize.x - 1) / blockSize.x, (numRows + blockSize.y - 1) / blockSize.y);

    //TODO: lanzar un kernel para separar la imagen RGB en diferentes canales de color
    separar_RGBA<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);

    cudaDeviceSynchronize(); 
	
    //TODO: Llame a su kernel de convolución aquí 3 veces, una para cada canal de color.    
    blur_Kernel<<<gridSize, blockSize>>>(  d_red,   d_redBlurred, numRows, numCols, d_filter, filterWidth);
    blur_Kernel<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
    blur_Kernel<<<gridSize, blockSize>>>( d_blue,  d_blueBlurred, numRows, numCols, d_filter, filterWidth);

    cudaDeviceSynchronize(); 

    // Ahora recombinamos tus resultados. Nos encargamos de lanzar este kernel
    recombinar_RGBA<<<gridSize, blockSize>>>(d_redBlurred, d_greenBlurred, d_blueBlurred, d_outputImageRGBA, numRows, numCols);
    cudaDeviceSynchronize(); 
}

void cargar_Imagen_crear_filtro_convolusion(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA, uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA, unsigned char **d_redBlurred, unsigned char **d_greenBlurred, unsigned char **d_blueBlurred, float **h_filter, int *filterWidth,const std::string &filename) {
    Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
    cvtColor(image, imageInputRGBA, CV_BGR2RGBA);
    imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);

    *h_inputImageRGBA  = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
    *h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);

    const size_t numPixels = numRows() * numCols();
    cudaMalloc(d_inputImageRGBA, sizeof(uchar4) * numPixels);
	cudaMalloc(d_outputImageRGBA, sizeof(uchar4) * numPixels);
	cudaMemset(*d_outputImageRGBA, 0, numPixels * sizeof(uchar4));

    cudaMemcpy(*d_inputImageRGBA, *h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);

    d_inputImageRGBA__  = *d_inputImageRGBA;
    d_outputImageRGBA__ = *d_outputImageRGBA;

    //ahora crearemos el filtro con el que convolveremos
    const int blurKernelWidth = 9; 
    const float blurKernelSigma = .02;

    *filterWidth = blurKernelWidth;

    //creamos y llenamos el filtro con el que convolveremos
    *h_filter = new float[blurKernelWidth * blurKernelWidth];
    h_filter__ = *h_filter;

    float filterSum = 0.f; // para la normalizacion

    for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
        for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
            float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
            (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
            filterSum += filterValue;
        }
    }

    float normalizationFactor = 1.f / filterSum;

    for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
        for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
            (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
        }
    }
	cudaMalloc(d_redBlurred,    sizeof(unsigned char) * numPixels);
    cudaMalloc(d_greenBlurred,  sizeof(unsigned char) * numPixels);
    cudaMalloc(d_blueBlurred,   sizeof(unsigned char) * numPixels);
    cudaMemset(*d_redBlurred,   0, sizeof(unsigned char) * numPixels);
    cudaMemset(*d_greenBlurred, 0, sizeof(unsigned char) * numPixels);
    cudaMemset(*d_blueBlurred,  0, sizeof(unsigned char) * numPixels);
}

int main(int argc, char **argv) {
	uchar4 *h_inputImageRGBA,  *d_inputImageRGBA;
    uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
    unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;
    float *h_filter;
    int    filterWidth;
	string input_file;
	input_file="lena.jpg";
    string output_file;
	output_file="inputBlur.jpg";

	cargar_Imagen_crear_filtro_convolusion(&h_inputImageRGBA, &h_outputImageRGBA, &d_inputImageRGBA, &d_outputImageRGBA, &d_redBlurred, &d_greenBlurred, &d_blueBlurred, &h_filter, &filterWidth, input_file);
	asign_memoria_copiar_Device(numRows(), numCols(), h_filter, filterWidth);

	filtro_a_realizar(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA, numRows(), numCols(), d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth);
    cudaDeviceSynchronize(); 

    const int numPixels = numRows() * numCols();
    cudaMemcpy(imageOutputRGBA.ptr<unsigned char>(0), d_outputImageRGBA__, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost);
    Mat imageOutputBGR;
    cvtColor(imageOutputRGBA, imageOutputBGR, CV_RGBA2BGR);
    imwrite(output_file.c_str(), imageOutputBGR);
    return 0;
}
