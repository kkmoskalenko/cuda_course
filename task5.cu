#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 16
#define BASE_TYPE double

int toMultiple(int a, int b)
{
    int mod = a % b;
    if (mod != 0)
    {
        mod = b - mod;
        return a + mod;
    }
    return a;
}

int main()
{
    // start, stop - for Kernel time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // количество строк и столбцов матрицы
    int Arows = 1000;
    int Acols = 2000;
    int Brows = Acols;
    int Bcols = 1500;

    Arows = toMultiple(Arows, BLOCK_SIZE);
    printf("Arows = %d\n", Arows);

    Acols = toMultiple(Acols, BLOCK_SIZE);
    printf("Acols = %d\n", Acols);

    Brows = toMultiple(Brows, BLOCK_SIZE);
    printf("Brows = %d\n", Brows);

    Bcols = toMultiple(Bcols, BLOCK_SIZE);
    printf("Bcols = %d\n", Bcols);

    size_t Asize = Arows * Acols * sizeof(BASE_TYPE);
    size_t Bsize = Brows * Bcols * sizeof(BASE_TYPE);
    size_t Csize = Arows * Bcols * sizeof(BASE_TYPE);

    BASE_TYPE *h_A = (BASE_TYPE *)malloc(Asize);
    BASE_TYPE *h_B = (BASE_TYPE *)malloc(Bsize);
    BASE_TYPE *h_C = (BASE_TYPE *)malloc(Csize);

    for (int i = 0; i < Arows * Acols; ++i)
        h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
    for (int i = 0; i < Brows * Bcols; ++i)
        h_B[i] = rand() / (BASE_TYPE)RAND_MAX;

    // Создаем временные массивы для column-major
    BASE_TYPE *h_A_col = (BASE_TYPE *)malloc(Asize);
    BASE_TYPE *h_B_col = (BASE_TYPE *)malloc(Bsize);
    BASE_TYPE *h_C_col = (BASE_TYPE *)malloc(Csize);

    // Конвертация h_A из row-major (Arows x Acols) в column-major
    for (int i = 0; i < Arows; i++) {
        for (int j = 0; j < Acols; j++) {
            h_A_col[j * Arows + i] = h_A[i * Acols + j];
        }
    }

    // Конвертация h_B из row-major (Brows x Bcols) в column-major
    for (int i = 0; i < Brows; i++) {
        for (int j = 0; j < Bcols; j++) {
            h_B_col[j * Brows + i] = h_B[i * Bcols + j];
        }
    }

    BASE_TYPE *d_A = NULL;
    cudaMalloc((void **)&d_A, Asize);
    BASE_TYPE *d_B = NULL;
    cudaMalloc((void **)&d_B, Bsize);
    BASE_TYPE *d_C = NULL;
    cudaMalloc((void **)&d_C, Csize);

    cudaMemcpy(d_A, h_A_col, Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_col, Bsize, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // cublasDgemm ожидает column-major
    // C = alpha * A * B + beta * C
    // Размеры: A (Arows x Acols), B (Acols x Bcols), C (Arows x Bcols)
    // lda = Arows, ldb = Brows (= Acols), ldc = Arows
    const double alpha = 1.0;
    const double beta = 0.0;

    cudaEventRecord(start, 0);
    cublasDgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                Arows, Bcols, Acols,
                &alpha,
                d_A, Arows,
                d_B, Brows,
                &beta,
                d_C, Arows);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float KernelTime;
    cudaEventElapsedTime(&KernelTime, start, stop);
    printf("KernelTime: %.2f milliseconds\n", KernelTime);

    cudaMemcpy(h_C_col, d_C, Csize, cudaMemcpyDeviceToHost);

    // Конвертируем h_C_col (column-major) обратно в row-major h_C
    for (int i = 0; i < Arows; i++) {
        for (int j = 0; j < Bcols; j++) {
            h_C[i * Bcols + j] = h_C_col[j * Arows + i];
        }
    }

    printf("Test STARTED\n");
    for (int i = 0; i < Arows; i++)
    {
        for (int j = 0; j < Bcols; j++)
        {
            BASE_TYPE sum = 0;
            for (int k = 0; k < Acols; k++)
                sum += h_A[i * Acols + k] * h_B[k * Bcols + j];
            if (fabs(h_C[i * Bcols + j] - sum) > 1e-3)
            {
                fprintf(stderr, "Result verification failed at element [%d, %d]!\n", i, j);
                printf("sum = %f, h_C[i * Bcols + j] = %f\n", sum, h_C[i * Bcols + j]);
                exit(EXIT_FAILURE);
            }
        }
    }
    printf("Test PASSED\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_A_col);
    free(h_B_col);
    free(h_C_col);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cublasDestroy(handle);

    return 0;
}