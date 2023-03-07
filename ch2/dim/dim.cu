#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>

__global__ void checkIndex(void) {
  printf("threadIdx:(%d ,%d, %d) blockIdx:(%d, %d, %d) blockDim(%d, %d, %d)"
         "grudDim:(%d ,%d, %d)\n",
         threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y,
         blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y,
         gridDim.z);
}

int main(int argc, char **argv) {
  int nElem = 6;

  dim3 block(3);
  dim3 grid((nElem + block.x - 1) / block.x);

  printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
  printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

  checkIndex<<<grid, block>>>();
  cudaDeviceReset();
  return (0);
}
