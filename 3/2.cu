#include <assert.h>
#include <stdio.h>

// A is n x m
// B is m
// C is n
__global__ void ker_matvecmul(const float *A, const float *B, float *C, int n,
                              int m) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  float dot = 0.0;
  for (int j = 0; j < m; ++j) {
    dot += A[idx * m + j] * B[j];
  }
  C[idx] = dot;
}

void matvecmul(const float *A, const float *B, float *C, int n, int m) {
  float *A_d, *B_d, *C_d;

  cudaMalloc((void **)&A_d, sizeof(float) * n * m);
  cudaMalloc((void **)&B_d, sizeof(float) * m);
  cudaMalloc((void **)&C_d, sizeof(float) * n);

  cudaMemcpy((void *)A_d, (const void *)A, sizeof(float) * n * m,
             cudaMemcpyHostToDevice);
  cudaMemcpy((void *)B_d, (const void *)B, sizeof(float) * m,
             cudaMemcpyHostToDevice);
  cudaMemcpy((void *)C_d, (const void *)C, sizeof(float) * n,
             cudaMemcpyHostToDevice);

  ker_matvecmul<<<ceil(n / 128.0), 128>>>(A_d, B_d, C_d, n, m);

  cudaMemcpy((void *)C, (const void *)C_d, sizeof(float) * n,
             cudaMemcpyDeviceToHost);

  cudaFree((void *)A_d);
  cudaFree((void *)B_d);
  cudaFree((void *)C_d);
}

int main(int argc, char *argv[]) {
  int n = 3;
  int m = 2;

  // 3x2
  float A[6] = {1.0, 2.0, 3.0, -4.0, -2.0, 3.0};

  // 2
  float B[2] = {2.0, 3.0};

  // 3
  float C[3] = {0};

  matvecmul(A, B, C, n, m);

  float C_exp[3] = {8.0, -6.0, 5.0};

  for (int i = 0; i < 3; ++i) {
    assert(abs(C[i] - C_exp[i]) < 0.01);
  }
  printf("OK\n");
}
