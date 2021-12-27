#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

double cpuMin(Mat& src) {

  vector<Mat> cpuMats;
  split(src, cpuMats);
  Mat blue = cpuMats[2];
  double min, max;
  minMaxLoc(blue, &min, &max);
  return min;

}

double gpuMin(cuda::GpuMat& src) {

  vector<cuda::GpuMat> gpuMats;
  cuda::split(src, gpuMats);
  cuda::GpuMat blue = gpuMats[2];
  double min, max;
  cuda::minMax(blue, &min, &max);
  return min;

}

int main() {
  Mat img = imread("ColorfulPic.jpg", IMREAD_COLOR);

  auto start = chrono::high_resolution_clock::now();
  double minCpu = cpuMin(img);
  auto end = chrono::high_resolution_clock::now();
  chrono::duration<double> diffCpu = end - start;

  cout << "CPU min: " << minCpu << endl;
  cout << "CPU took: " << setw(7) << diffCpu.count() << "sec" << endl;

  cuda::GpuMat src;
  src.upload(img);

  start = chrono::high_resolution_clock::now();
  double minGpu = gpuMin(src);
  end = chrono::high_resolution_clock::now();
  chrono::duration<double> diffGpu = end - start;

  cout << "GPU min: " << minGpu << endl;
  cout << "GPU CUDA took: " << setw(7) << diffGpu.count() << "sec" << endl;
}