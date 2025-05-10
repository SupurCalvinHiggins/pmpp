#include <assert.h>
#include <stdio.h>

#define RM_IDX(row, col, width) (row * width + col)

// A is n x k
// B is k x m
// C is n x m
__global__ void ker_matmul(const float *A, const float *B, float *C, int n,
                           int k, int m) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= m) {
    return;
  }

  for (int row = 0; row < n; ++row) {
    float dot = 0.0;
    for (int j = 0; j < k; ++j) {
      dot += A[RM_IDX(row, j, k)] * B[RM_IDX(j, col, m)];
    }
    C[RM_IDX(row, col, m)] = dot;
  }
}

void matmul(const float *A, const float *B, float *C, int n, int k, int m) {
  float *A_d, *B_d, *C_d;

  cudaMalloc((void **)&A_d, sizeof(float) * n * k);
  cudaMalloc((void **)&B_d, sizeof(float) * k * m);
  cudaMalloc((void **)&C_d, sizeof(float) * n * m);

  cudaMemcpy((void *)A_d, (const void *)A, sizeof(float) * n * k,
             cudaMemcpyHostToDevice);
  cudaMemcpy((void *)B_d, (const void *)B, sizeof(float) * k * m,
             cudaMemcpyHostToDevice);
  cudaMemcpy((void *)C_d, (void *)C, sizeof(float) * n * m,
             cudaMemcpyHostToDevice);

  ker_matmul<<<ceil(m / 128.0), 128>>>(A_d, B_d, C_d, n, k, m);

  cudaMemcpy((void *)C, (void *)C_d, sizeof(float) * n * m,
             cudaMemcpyDeviceToHost);

  cudaFree((void *)A_d);
  cudaFree((void *)B_d);
  cudaFree((void *)C_d);
}

int main(int argc, char *argv[]) {
  int n = 3;
  int k = 2;
  int m = 4;

  // 3x2
  float A[6] = {1.0, 2.0, 3.0, -4.0, -2.0, 3.0};

  // 2x4
  float B[8] = {
      2.0, 3.0, -3.0, -4.0, -5.0, 1.0, 0.0, 2.0,
  };

  // 3x4
  float C[12] = {0};

  matmul(A, B, C, n, k, m);

  float C_exp[12] = {
      -8, 5, -3, 0, 26, 5, -9, -20, -19, -3, 6, 14,
  };

  for (int i = 0; i < 12; ++i) {
    assert(abs(C[i] - C_exp[i]) < 0.01);
  }
  printf("OK\n");
}
