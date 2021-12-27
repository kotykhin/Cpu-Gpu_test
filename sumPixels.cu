#include <CImg.h>
#include <math.h>
#include <stdlib.h>
#include <cuda.h>

using namespace cimg_library;

void sum_cpu(float* sum, CImg< unsigned char >* inputImage, int height, int width)
{
    for ( int y = 0; y < height; y++ ) {
      for ( int x = 0; x < width; x++ ) {
        float r = static_cast< float >(inputImage[(y * width) + x]);

        sum += r;
        }
     }
}

 __global__ void sum_gpu(float* sum, CImg< unsigned char >* inputImage, int height, int width)
{
    unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if ((x < width) && (y < height)){ 
      float r = static_cast< float >(inputImage[(y * width) + x]);

      sum += r;
      }
}

int main(int argc, char *argv[]){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float millisecondsCpu = 0;
  float millisecondsGpu = 0;
  
  CImg<unsigned char> img(640,400,1,3);

  float *img, *imgf, *kernel;
  
  img = (float*)malloc(Nx*Ny*sizeof(float));
  imgf = (float*)malloc(Nx*Ny*sizeof(float));
  kernel = (float*)malloc(kernel_size*kernel_size*sizeof(float));  
  
  
  float *d_img, *d_imgf, *d_kernel;
  
  cudaMalloc(&d_img,Nx*Ny*sizeof(float));
  cudaMalloc(&d_imgf,Nx*Ny*sizeof(float));
  cudaMalloc(&d_kernel,kernel_size*kernel_size*sizeof(float));
  
  load_image(finput, Nx, Ny, img);
  calculate_kernel(kernel_size, sigma, kernel);

  cudaMemcpy(d_img, img, Nx*Ny*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel,kernel, kernel_size*kernel_size*sizeof(float),cudaMemcpyHostToDevice);

  Nblocks = Ny - (kernel_size-1);
  Nthreads = Nx - (kernel_size-1);
  
  cudaEventRecord(start);
  sum_cpu(img, kernel, imgf, Nx, Ny, kernel_size);
  cudaEventRecord(stop);
  cudaEventElapsedTime(&millisecondsCpu, start, stop);

  printf("\n");
  printf("Ellapsed Time (CPU): %16.10f ms\n", millisecondsCpu);
  printf("\n");

  cudaEventRecord(start);
  conv_img_gpu<<<Nblocks, Nthreads, kernel_size*kernel_size*sizeof(float)>>>(d_img, d_kernel, d_imgf, Nx, Ny, kernel_size);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventElapsedTime(&millisecondsGpu, start, stop);
  
  cudaMemcpy(imgf, d_imgf, Nx*Ny*sizeof(float), cudaMemcpyDeviceToHost);
  save_image(foutput, Nx, Ny, imgf);
  
  printf("\n");
  printf("Ellapsed Time (GPU): %16.10f ms\n", millisecondsGpu);
  printf("\n");
  
  
  free(img);
  free(imgf);
  free(kernel);

  cudaFree(d_img);
  cudaFree(d_imgf);
  cudaFree(d_kernel);
}