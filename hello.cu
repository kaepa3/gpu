#include <iostream>
#include <stdio.h>
#include <time.h>
#include <type_traits>

__global__ void gpu_function(float *d_x, float *d_y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  d_y[i] = sin(d_x[i]) * sin(d_x[i]) + cos(d_x[i]) * cos(d_x[i]);
}

void cpu_function(int n, float *x, float *y) {
  for (int i = 0; i < n; i++) {
    y[i] = sin(x[i]) * sin(x[i]) + cos(x[i]) * cos(x[i]);
  }
}

int main() {

  bool gpu = true;
  int N = 10000000;
  float *host_x, *host_y, *dev_x, *dev_y;
  host_x = (float *)malloc(N * sizeof(float));
  host_y = (float *)malloc(N * sizeof(float));

  for (int i = 0; i < N; i++) {
    host_x[i] = rand();
  }

  int start = clock();
  if (gpu) {
    cudaMalloc(&dev_x, N * sizeof(float));
    cudaMalloc(&dev_y, N * sizeof(float));

    cudaMemcpy(dev_x, host_x, N * sizeof(float), cudaMemcpyHostToDevice);
    gpu_function<<<(N + 255) / 256, 256>>>(dev_x, dev_y);
    cudaMemcpy(host_y, dev_y, N * sizeof(float), cudaMemcpyDeviceToHost);

  } else {
    cpu_function(N, host_x, host_y);
  }

  // 計算が正しく行われているか確認
  float sum = 0.0f;
  for (int j = 0; j < N; j++) {
    sum += host_y[j];
  }
  std::cout << sum << std::endl;

  int end = clock();

  std::cout << end - start << "[ms]";
  return 0;
}
