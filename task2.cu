#include <stdio.h>
#include <stdlib.h>
#include "png_utils.h"
#include <cuda_runtime.h>
#include <time.h>

#define SIGMA 25.0f
#define FILTER_SIZE 31
#define BLOCK_SIZE 16

__global__ void convolutionShared(unsigned char* input, unsigned char* output, int width, int height, float* filter, int filterWidth) {
    extern __shared__ unsigned char sharedMem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;
    int radius = filterWidth / 2;

    int sharedWidth = blockDim.x + 2 * radius;
    int sharedX = tx + radius;
    int sharedY = ty + radius;

    for (int c = 0; c < 3; c++) {
        int inputX = min(max(col, 0), width - 1);
        int inputY = min(max(row, 0), height -1);
        sharedMem[(sharedY * sharedWidth + sharedX) * 3 + c] = input[(inputY * width + inputX) * 3 + c];
    }

    for(int dy = -radius; dy <= radius; dy++) {
        for(int dx = -radius; dx <= radius; dx++) {
            int sharedMemX = sharedX + dx;
            int sharedMemY = sharedY + dy;
            int globalX = min(max(col + dx, 0), width -1);
            int globalY = min(max(row + dy, 0), height -1);
            for(int c = 0; c < 3; c++)
                sharedMem[(sharedMemY * sharedWidth + sharedMemX) * 3 + c] = input[(globalY * width + globalX) * 3 + c];
        }
    }
    __syncthreads();

    if(row < height && col < width) {
        for(int c = 0; c < 3; c++) {
            float sum = 0.0f;
            for(int fy = 0; fy < filterWidth; fy++) {
                for(int fx = 0; fx < filterWidth; fx++) {
                    int imageRow = sharedY + fy - radius;
                    int imageCol = sharedX + fx - radius;
                    float imageVal = sharedMem[(imageRow * sharedWidth + imageCol) * 3 + c];
                    float filterVal = filter[fy * filterWidth + fx];
                    sum += filterVal * imageVal;
                }
            }
            output[(row * width + col) * 3 + c] = min(max(int(sum), 0), 255);
        }
    }
}

__global__ void convolutionTexture(cudaTextureObject_t texObj, unsigned char* output, int width, int height, float* filter, int filterWidth) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = filterWidth / 2;

    if (row < height && col < width) {
        float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
        for (int fy = -radius; fy <= radius; fy++) {
            for (int fx = -radius; fx <= radius; fx++) {
                int x = min(max(col + fx, 0), width - 1);
                int y = min(max(row + fy, 0), height - 1);
                uchar4 pixel = tex2D<uchar4>(texObj, x, y);
                float filterVal = filter[(fy + radius) * filterWidth + (fx + radius)];
                sumR += pixel.x * filterVal;
                sumG += pixel.y * filterVal;
                sumB += pixel.z * filterVal;
            }
        }
        output[(row * width + col) * 3 + 0] = min(max(int(sumR), 0), 255);
        output[(row * width + col) * 3 + 1] = min(max(int(sumG), 0), 255);
        output[(row * width + col) * 3 + 2] = min(max(int(sumB), 0), 255);
    }
}

void generateGaussianFilter(float* filter, int filterSize, float sigma) {
    int radius = filterSize / 2;
    float sum = 0.0f;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float exponent = -(x * x + y * y) / (2 * sigma * sigma);
            float value = exp(exponent);
            filter[(y + radius) * filterSize + (x + radius)] = value;
            sum += value;
        }
    }
    for (int i = 0; i < filterSize * filterSize; i++) {
        filter[i] /= sum;
    }
}

int main() {
    struct timespec mainStart, mainEnd;
    clock_gettime(CLOCK_MONOTONIC, &mainStart);

    int width, height;
    struct timespec readStart, readEnd;
    clock_gettime(CLOCK_MONOTONIC, &readStart);
    unsigned char* h_inputImage = readPNG("input.png", &width, &height);
    clock_gettime(CLOCK_MONOTONIC, &readEnd);

    if (!h_inputImage) {
        printf("Не удалось прочитать входное изображение\n");
        return -1;
    }
    size_t imageSize = width * height * 3 * sizeof(unsigned char);

    unsigned char *d_inputImage, *d_outputImageShared, *d_outputImageTexture;
    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImageShared, imageSize);
    cudaMalloc(&d_outputImageTexture, imageSize);

    struct timespec dataTransferStart, dataTransferEnd;
    clock_gettime(CLOCK_MONOTONIC, &dataTransferStart);
    cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_MONOTONIC, &dataTransferEnd);

    int filterWidth = FILTER_SIZE;
    size_t filterSize = filterWidth * filterWidth * sizeof(float);
    float* h_filter = (float*)malloc(filterSize);
    float sigma = SIGMA;
    generateGaussianFilter(h_filter, filterWidth, sigma);

    float* d_filter;
    cudaMalloc(&d_filter, filterSize);
    cudaMemcpy(d_filter, h_filter, filterSize, cudaMemcpyHostToDevice);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    unsigned char* h_inputImageRGBA = (unsigned char*)malloc(width * height * 4);
    for(int i = 0; i < width * height; i++) {
        h_inputImageRGBA[i * 4 + 0] = h_inputImage[i * 3 + 0];
        h_inputImageRGBA[i * 4 + 1] = h_inputImage[i * 3 + 1];
        h_inputImageRGBA[i * 4 + 2] = h_inputImage[i * 3 + 2];
        h_inputImageRGBA[i * 4 + 3] = 255;
    }

    struct timespec textureTransferStart, textureTransferEnd;
    clock_gettime(CLOCK_MONOTONIC, &textureTransferStart);
    cudaMemcpyToArray(cuArray, 0, 0, h_inputImageRGBA, width * height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_MONOTONIC, &textureTransferEnd);

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    size_t sharedMemSize = (blockSize.x + FILTER_SIZE - 1) * (blockSize.y + FILTER_SIZE - 1) * 3 * sizeof(unsigned char);

    cudaEvent_t startEvent, stopEvent;
    float sharedTime, textureTime;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent);
    convolutionShared<<<gridSize, blockSize, sharedMemSize>>>(d_inputImage, d_outputImageShared, width, height, d_filter, FILTER_SIZE);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&sharedTime, startEvent, stopEvent);

    cudaEventRecord(startEvent);
    convolutionTexture<<<gridSize, blockSize>>>(texObj, d_outputImageTexture, width, height, d_filter, FILTER_SIZE);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&textureTime, startEvent, stopEvent);

    unsigned char* h_outputImageShared = (unsigned char*)malloc(imageSize);
    unsigned char* h_outputImageTexture = (unsigned char*)malloc(imageSize);
    cudaMemcpy(h_outputImageShared, d_outputImageShared, imageSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outputImageTexture, d_outputImageTexture, imageSize, cudaMemcpyDeviceToHost);

    struct timespec writeStart, writeEnd;
    clock_gettime(CLOCK_MONOTONIC, &writeStart);
    writePNG("output_shared.png", width, height, h_outputImageShared);
    writePNG("output_texture.png", width, height, h_outputImageTexture);
    clock_gettime(CLOCK_MONOTONIC, &writeEnd);

    printf("Shared Memory Kernel Execution Time: %f ms\n", sharedTime);
    printf("Texture Memory Kernel Execution Time: %f ms\n", textureTime);

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(d_inputImage);
    cudaFree(d_outputImageShared);
    cudaFree(d_outputImageTexture);
    cudaFree(d_filter);
    free(h_filter);
    free(h_inputImage);
    free(h_inputImageRGBA);
    free(h_outputImageShared);
    free(h_outputImageTexture);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    double dataTransferTime = (dataTransferEnd.tv_sec - dataTransferStart.tv_sec) * 1000.0 +
                              (dataTransferEnd.tv_nsec - dataTransferStart.tv_nsec) / 1000000.0;
    printf("Data Transfer to GPU Time: %f ms\n", dataTransferTime);

    double textureTransferTime = (textureTransferEnd.tv_sec - textureTransferStart.tv_sec) * 1000.0 +
                                 (textureTransferEnd.tv_nsec - textureTransferStart.tv_nsec) / 1000000.0;
    printf("Texture Transfer to GPU Time: %f ms\n", textureTransferTime);

    double readTime = (readEnd.tv_sec - readStart.tv_sec) * 1000.0 +
                      (readEnd.tv_nsec - readStart.tv_nsec) / 1000000.0;
    printf("PNG Read Time: %f ms\n", readTime);

    double writeTime = (writeEnd.tv_sec - writeStart.tv_sec) * 1000.0 +
                       (writeEnd.tv_nsec - writeStart.tv_nsec) / 1000000.0;
    printf("PNG Write Time: %f ms\n", writeTime);

    clock_gettime(CLOCK_MONOTONIC, &mainEnd);
    double mainTime = (mainEnd.tv_sec - mainStart.tv_sec) * 1000.0 +
                      (mainEnd.tv_nsec - mainStart.tv_nsec) / 1000000.0;
    printf("Total Execution Time (main): %f ms\n", mainTime);

    return 0;
}