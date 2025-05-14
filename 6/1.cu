#include <assert.h>
#include <stdio.h>

// A is n x n
// B is n x n
// C is n x n

#define TILE_WIDTH 32

__global__ void ker_matmul(const float *A, const float *B, float *C,
                           int width) {

  __shared__ float At[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Bt[TILE_WIDTH][TILE_WIDTH];

  float acc[TILE_WIDTH];
  for (int r = 0; r < TILE_WIDTH; ++r) {
    acc[r] = 0.0;
  }

  const int tile_col = blockIdx.x;
  const int tile_row = blockIdx.y;
  const int elem_col = threadIdx.x;

  const int tile_row_off = tile_row * TILE_WIDTH;
  const int tile_col_off = tile_col * TILE_WIDTH * TILE_WIDTH;
  const int elem_col_off = elem_col;

  for (int p = 0; p < width / TILE_WIDTH; ++p) {
    // Copy data into At and Bt.
    // At is the tile at (row, p)
    // Bt is the tile at (p, col)
    for (int elem_row = 0; elem_row < TILE_WIDTH; ++elem_row) {
      const int elem_row_off = elem_row * TILE_WIDTH;
      At[elem_row][elem_col] = A[tile_row_off + (p * TILE_WIDTH * TILE_WIDTH) +
                                 elem_row_off + elem_col_off];
      Bt[elem_row][elem_col] = B[(p * TILE_WIDTH * TILE_WIDTH) + tile_col_off +
                                 elem_row_off + elem_col_off];
    }
    __syncthreads();

    // Accumulate into acc.
    for (int elem_row = 0; elem_row < TILE_WIDTH; ++elem_row) {
      for (int i = 0; i < TILE_WIDTH; ++i) {
        acc[elem_row] += At[elem_row][i] * Bt[i][elem_col];
      }
    }
    __syncthreads();
  }

  for (int elem_row = 0; elem_row < TILE_WIDTH; ++elem_row) {
    const int elem_row_off = elem_row * TILE_WIDTH;
    C[tile_row_off + tile_col_off + elem_row_off + elem_col_off] =
        acc[elem_row];
  }
}

void matmul(const float *A, const float *B, float *C, int width) {
  float *A_d, *B_d, *C_d;

  cudaMalloc(&A_d, width * width * sizeof(float));
  cudaMalloc(&B_d, width * width * sizeof(float));
  cudaMalloc(&C_d, width * width * sizeof(float));

  cudaMemcpy(A_d, A, width * width * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, width * width * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(C_d, C, width * width * sizeof(float), cudaMemcpyHostToDevice);

  const dim3 gridDim(width, width / TILE_WIDTH, 1);
  const dim3 blockDim(TILE_WIDTH, 1, 1);
  ker_matmul<<<gridDim, blockDim>>>(A_d, B_d, C_d, width);

  cudaMemcpy(C, C_d, width * width * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

int main(int argc, char *argv[]) {
  const int width = 8 * TILE_WIDTH;

  float A[width][width];
  float B[width][width];
  float C[width][width];

  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < width; ++j) {
      A[i][j] = 2.0 * i * width - 3.0 * j;
      B[i][j] = -j * width + 5.0 * i;
      C[i][j] = 0.0;
    }
  }

  matmul(A, B, C, width);

  for (int i = 0; i < width; ++i) {
    for (int j = 0; j < width; ++j) {
      int acc = 0.0;
      for (int k = 0; k < width; ++k) {
        acc += A[i][k] * B[k][j];
      }
      printf("%g %g", C[i][j], acc);
      assert(abs(C[i][j] - acc) < 0.001);
    }
  }
  printf("OK\n");
}
