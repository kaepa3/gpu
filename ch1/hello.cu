#include <stdio.h>

__global__ void helloFromGPU() {

  int idx = threadIdx.x;
  printf("hello world from gpu %d\n", idx);
}

int main(int argc, char **argv) {
  printf("hello! world from cpu\n");

  helloFromGPU<<<1, 10>>>();
  cudaDeviceSynchronize();
  return 0;
}
