#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      printf("Error: %s:%d", __FILE__, __LINE__);                              \
      printf("code:%d, reson: %s\n", error, cudaGetErrorString(error));        \
      exit(1);                                                                 \
    }                                                                          \
  }

void printMatrix(int *C, const int nx, const int ny) {
  int *ic = C;
  printf("\nMatrix: (%d.%d)\n", nx, ny);
  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++) {
      printf("%3d", ic[ix]);
    }
    ic += nx;
    printf("\n");
  }
  printf("\n");
  return;
}

__global__ void printThreadIndex(int *A, const int nx, const int ny) {
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy * nx + ix;
  printf("thread_id(%d,%d) block_id(%d,%d) coordinate (%d,%d)"
         "global index %2d ival %2d\n",
         threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny) {
  float *ia = A;
  float *ib = B;
  float *ic = C;
  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++) {
      ic[ix] = ia[ix] + ib[ix];
    }
    ia += nx;
    ib += nx;
    ic += nx;
  }
  return;
}
__global__ void sumMatrixOnGpu1D(float *MatA, float *MatB, float *MatC, int nx,
                                 int ny) {
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  if (ix < nx) {
    for (int iy = 0; iy < ny; iy++) {
      int idx = iy * nx + ix;
      MatC[idx] = MatA[idx] + MatB[idx];
    }
  }
}

__global__ void sumMatrixOnGpu2D(float *MatA, float *MatB, float *MatC, int nx,
                                 int ny) {
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy * nx + ix;
  if (ix < nx && iy < ny) {
    MatC[idx] = MatA[idx] + MatB[idx];
  }
}

void initialData(float *ip, const int size) {
  int i;
  for (i = 0; i < size; i++) {
    ip[i] = (float)(rand() & 0xFF) / 10.0f;
  }
  return;
}

int main(int argc, char **argv) {
  printf("%s Starting....\n", argv[0]);

  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Using Device %d: %s \n", dev, deviceProp.name);

  int nx = 1 << 14;
  int ny = 1 << 14;
  int nxy = nx * ny;
  int nBytes = nxy * sizeof(float);

  // host
  float *h_A, *h_B, *hostRef, *gpuRef;
  h_A = (float *)malloc(nBytes);
  h_B = (float *)malloc(nBytes);
  hostRef = (float *)malloc(nBytes);
  gpuRef = (float *)malloc(nBytes);

  // init
  initialData(h_A, nxy);
  initialData(h_B, nxy);
  memset(hostRef, 0, nBytes);
  memset(gpuRef, 0, nBytes);

  DWORD start = GetTickCount();
  sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
  DWORD end = GetTickCount();
  DWORD elaps = end - start;
  printf("sumMatrixOnHost elasped %lu (sec)\n", elaps);

  // device
  float *d_MatA, *d_MatB, *d_MatC;
  CHECK(cudaMalloc((void **)&d_MatA, nBytes));
  CHECK(cudaMalloc((void **)&d_MatB, nBytes));
  CHECK(cudaMalloc((void **)&d_MatC, nBytes));

  CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_MatC, h_B, nBytes, cudaMemcpyHostToDevice));

  int dimx = 32;
  int dimy = 16;
  dim3 block(dimx, dimy);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
  start = GetTickCount();
  sumMatrixOnGpu2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
  CHECK(cudaDeviceSynchronize());
  end = GetTickCount();
  elaps = end - start;
  printf("sumMatrixOnGpu2D <<<(%d,%d),(%d,%d)>>> elasped %lu (sec)\n", grid.x,
         grid.y, block.x, block.y, elaps);

  free(h_A);
  free(h_B);
  free(hostRef);
  free(gpuRef);

  CHECK(cudaFree(d_MatA));
  CHECK(cudaFree(d_MatB));
  CHECK(cudaFree(d_MatC));

  CHECK(cudaDeviceReset());
  return (0);
}
