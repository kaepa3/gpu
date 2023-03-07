#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void sumArrayOnHost(float *A, float *B, float *C, const int N) {
  for (int idx = 0; idx < N; idx++) {
    C[idx] = A[idx] + B[idx];
  }
}

void initialData(float *ip, int size) {
  time_t t;
  srand((unsigned int)time(&t));

  for (int i = 0; i < size; i++) {
    ip[i] = (float)(rand() & 0xFF) / 10.0f;
  }
  return;
}

int main(int argc, char **argv) {
  int nElen = 1024;
  size_t nBytes = nElen * sizeof(float);
  float *h_A, *h_B, *h_C;
  h_A = (float *)malloc(nBytes);
  h_B = (float *)malloc(nBytes);
  h_C = (float *)malloc(nBytes);

  float *d_A, *d_B, *d_C;
  cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToHost);
  cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToHost);

  sumArrayOnHost(h_A, h_B, h_C, nElen);

  cudaMemcpy(d_C, h_C, nBytes, cudaMemcpyDeviceToHost);
  free(h_A);
  free(h_B);
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return (0);
}
