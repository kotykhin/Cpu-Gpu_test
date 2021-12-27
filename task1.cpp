#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

double cpuSum(Mat& src) {

  vector<Mat> cpuMats;
  split(src, cpuMats);
  Mat blue = cpuMats[2];
  double pixelsum = sum(blue).val[0];
  return pixelsum;

}


double gpuSum(cuda::GpuMat& src) {

  vector<cuda::GpuMat> gpuMats;
  cuda::split(src, gpuMats);
  cuda::GpuMat blue = gpuMats[2];
  double sum = cuda::sum(blue).val[0];
  return sum;
  
}

int main() {
  Mat img = imread("ColorfulPic.jpg", IMREAD_COLOR);

  auto start = chrono::high_resolution_clock::now();
  double sumCpu = cpuSum(img);
  auto end = chrono::high_resolution_clock::now();
  chrono::duration<double> diffCpu = end - start;

  cout << "CPU sum: " << sumCpu << endl;
  cout << "CPU took: " << setw(7) << diffCpu.count() << "sec" << endl;

  cuda::GpuMat src;
  src.upload(img);

  start = chrono::high_resolution_clock::now();
  double sumGpu = gpuSum(src);
  end = chrono::high_resolution_clock::now();
  chrono::duration<double> diffGpu = end - start;

  cout << "GPU sum: " << sumGpu << endl;
  cout << "GPU CUDA took: " << setw(7) << diffGpu.count() << "sec" << endl;
}