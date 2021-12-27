
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

#define pi 3.14159265359

void load_image(char* fname, int Nx, int Ny, float* img) {
  FILE* fp;

  fp = fopen(fname, "r");

  for (int i = 0; i < Ny; i++) {
    for (int j = 0; j < Nx; j++)
      fscanf(fp, "%f ", &img[i * Nx + j]);
    fscanf(fp, "\n");
  }

  fclose(fp);
}


void save_image(char* fname, int Nx, int Ny, float* img) {
  FILE* fp;

  fp = fopen(fname, "w");

  for (int i = 0; i < Ny; i++) {
    for (int j = 0; j < Nx; j++)
      fprintf(fp, "%10.3f ", img[i * Nx + j]);
    fprintf(fp, "\n");
  }

  fclose(fp);
}


void calculate_kernel(int kernel_size, float sigma, float* kernel) {

  int Nk2 = kernel_size * kernel_size;
  float x, y, center;

  center = (kernel_size - 1) / 2.0;

  for (int i = 0; i < Nk2; i++) {
    x = (float)(i % kernel_size) - center;
    y = (float)(i / kernel_size) - center;
    kernel[i] = -(1.0 / pi * pow(sigma, 4)) * (1.0 - 0.5 * (x * x + y * y) / (sigma * sigma)) * exp(-0.5 * (x * x + y * y) / (sigma * sigma));
  }

}

void conv_img_cpu(float* img, float* kernel, float* imgf, int Nx, int Ny, int kernel_size)
{

  float sum = 0;
  int center = (kernel_size - 1) / 2;
  int ii, jj;

  for (int i = center; i < (Ny - center); i++)
    for (int j = center; j < (Nx - center); j++) {
      sum = 0;
      for (int ki = 0; ki < kernel_size; ki++)
        for (int kj = 0; kj < kernel_size; kj++) {
          ii = kj + j - center;
          jj = ki + i - center;
          sum += img[jj * Nx + ii] * kernel[ki * kernel_size + kj];
        }
      imgf[i * Nx + j] = sum;
    }
}


__global__ void conv_img_gpu(float* img, float* kernel, float* imgf, int Nx, int Ny, int kernel_size)
{

  int tid = threadIdx.x;

  int iy = blockIdx.x + (kernel_size - 1) / 2;

  int ix = threadIdx.x + (kernel_size - 1) / 2;

  int idx = iy * Nx + ix;

  int K2 = kernel_size * kernel_size;

  int center = (kernel_size - 1) / 2;
  int ii, jj;
  float sum = 0.0;

  extern __shared__ float sdata[];

  if (tid < K2)
    sdata[tid] = kernel[tid];

  __syncthreads();

  if (idx < Nx * Ny) {
    for (int ki = 0; ki < kernel_size; ki++)
      for (int kj = 0; kj < kernel_size; kj++) {
        ii = kj + ix - center;
        jj = ki + iy - center;
        sum += img[jj * Nx + ii] * sdata[ki * kernel_size + kj];
      }

    imgf[idx] = sum;
  }
}

int main(int argc, char* argv[]) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float millisecondsCpu = 0;
  float millisecondsGpu = 0;
  int Nx, Ny;
  int kernel_size;
  float sigma;
  char finput[256], foutput[256];
  int Nblocks, Nthreads;


  sprintf(finput, "input.dat");
  sprintf(foutput, "output.dat");

  Nx = 256;
  Ny = 256;

  kernel_size = 5;
  sigma = 0.8;


  float* img, * imgf, * kernel;

  img = (float*)malloc(Nx * Ny * sizeof(float));
  imgf = (float*)malloc(Nx * Ny * sizeof(float));
  kernel = (float*)malloc(kernel_size * kernel_size * sizeof(float));


  float* d_img, * d_imgf, * d_kernel;

  cudaMalloc(&d_img, Nx * Ny * sizeof(float));
  cudaMalloc(&d_imgf, Nx * Ny * sizeof(float));
  cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float));

  load_image(finput, Nx, Ny, img);
  calculate_kernel(kernel_size, sigma, kernel);

  cudaMemcpy(d_img, img, Nx * Ny * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

  Nblocks = Ny - (kernel_size - 1);
  Nthreads = Nx - (kernel_size - 1);

  auto start_time = std::chrono::high_resolution_clock::now();
  conv_img_cpu(img, kernel, imgf, Nx, Ny, kernel_size);
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diffCpu = end_time - start_time;

  printf("\n");
  printf("Ellapsed Time (CPU): %16.10f ms\n", diffCpu);
  printf("\n");

  start_time = std::chrono::high_resolution_clock::now();
  conv_img_gpu << <Nblocks, Nthreads, kernel_size* kernel_size * sizeof(float) >> > (d_img, d_kernel, d_imgf, Nx, Ny, kernel_size);
  end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diffGpu = end_time - start_time;
  cudaDeviceSynchronize();

  cudaMemcpy(imgf, d_imgf, Nx * Ny * sizeof(float), cudaMemcpyDeviceToHost);
  save_image(foutput, Nx, Ny, imgf);

  printf("\n");
  printf("Ellapsed Time (GPU): %16.10f ms\n", diffGpu);
  printf("\n");


  free(img);
  free(imgf);
  free(kernel);

  cudaFree(d_img);
  cudaFree(d_imgf);
  cudaFree(d_kernel);
}