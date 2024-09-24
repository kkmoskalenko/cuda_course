#include <cmath>
#include <iostream>

constexpr size_t N = 1e9;

constexpr auto threadsPerBlock = dim3(256);
constexpr auto blocksPerGrid = dim3((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

template<typename T>
__global__ void sin_kernel(T *arr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        arr[idx] = sin((idx % 360) * M_PI / 180);
    }
}

template<typename T>
__global__ void sinf_kernel(T *arr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        arr[idx] = sinf((idx % 360) * M_PI / 180);
    }
}

template<typename T>
__global__ void __sinf_kernel(T *arr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        arr[idx] = __sinf((idx % 360) * M_PI / 180);
    }
}

template<typename T>
double calculate_error(const T *h_arr) {
    double err = 0;
    for (int i = 0; i < N; i++) {
        const double expected = sin((i % 360) * M_PI / 180);
        err += abs(expected - h_arr[i]);
    }
    return err / N;
}

template<typename KernelFunc, typename Num>
void executeKernel(KernelFunc kernel, Num *d_arr, Num *h_arr, const char *kernelName) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_arr, d_arr, sizeof(Num) * N, cudaMemcpyDeviceToHost);
    std::cout << kernelName << ": error = " << calculate_error(h_arr)
            << ", time = " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template<typename Num>
void benchmark() {
    Num *d_arr, *h_arr;

    cudaMalloc(&d_arr, sizeof(Num) * N);
    h_arr = new Num[N];

    executeKernel(sin_kernel<Num>, d_arr, h_arr, "sin");
    executeKernel(sinf_kernel<Num>, d_arr, h_arr, "sinf");
    executeKernel(__sinf_kernel<Num>, d_arr, h_arr, "__sinf");

    cudaFree(d_arr);
    delete[] h_arr;
}

int main() {
    std::cout << "float benchmark:" << std::endl;
    benchmark<float>();

    std::cout << std::endl << "double benchmark:" << std::endl;
    benchmark<double>();

    return 0;
}
