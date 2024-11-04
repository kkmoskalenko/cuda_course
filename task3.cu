#include <stdio.h>
#include <stdlib.h>
#include "png_utils.h"
#include <cuda_runtime.h>

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

void processOnGPU(int deviceID, int numGPUs, unsigned char* h_inputImage, unsigned char* h_outputImageShared, unsigned char* h_outputImageTexture, 
                  int width, int height, float* h_filter, int filterWidth, float sigma, float* sharedTime, float* textureTime) {
    cudaSetDevice(deviceID);

    int overlap = filterWidth / 2;
    int segmentHeight = height / numGPUs;
    int yOffset = segmentHeight * deviceID;

    if (deviceID == numGPUs - 1) {
        segmentHeight += height % numGPUs;
    }

    int startRow = (deviceID == 0) ? 0 : yOffset - overlap;
    int endRow = (deviceID == numGPUs - 1) ? height : yOffset + segmentHeight + overlap;
    int segmentHeightWithOverlap = endRow - startRow;

    size_t segmentSizeWithOverlap = width * segmentHeightWithOverlap * 3 * sizeof(unsigned char);
    unsigned char *d_inputImage, *d_outputImageShared, *d_outputImageTexture;
    cudaMalloc(&d_inputImage, segmentSizeWithOverlap);
    cudaMalloc(&d_outputImageShared, segmentSizeWithOverlap);
    cudaMalloc(&d_outputImageTexture, segmentSizeWithOverlap);

    cudaMemcpy(d_inputImage, h_inputImage + startRow * width * 3, segmentSizeWithOverlap, cudaMemcpyHostToDevice);

    float* d_filter;
    size_t filterSize = filterWidth * filterWidth * sizeof(float);
    cudaMalloc(&d_filter, filterSize);
    cudaMemcpy(d_filter, h_filter, filterSize, cudaMemcpyHostToDevice);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, segmentHeightWithOverlap);

    unsigned char* h_inputImageSegmentRGBA = (unsigned char*)malloc(width * segmentHeightWithOverlap * 4);
    for (int i = 0; i < width * segmentHeightWithOverlap; i++) {
        h_inputImageSegmentRGBA[i * 4 + 0] = h_inputImage[(startRow * width + i) * 3 + 0];
        h_inputImageSegmentRGBA[i * 4 + 1] = h_inputImage[(startRow * width + i) * 3 + 1];
        h_inputImageSegmentRGBA[i * 4 + 2] = h_inputImage[(startRow * width + i) * 3 + 2];
        h_inputImageSegmentRGBA[i * 4 + 3] = 255;
    }

    cudaMemcpyToArray(cuArray, 0, 0, h_inputImageSegmentRGBA, width * segmentHeightWithOverlap * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);

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
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (segmentHeightWithOverlap + blockSize.y - 1) / blockSize.y);
    size_t sharedMemSize = (blockSize.x + FILTER_SIZE - 1) * (blockSize.y + FILTER_SIZE - 1) * 3 * sizeof(unsigned char);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent);
    convolutionShared<<<gridSize, blockSize, sharedMemSize>>>(d_inputImage, d_outputImageShared, width, segmentHeightWithOverlap, d_filter, FILTER_SIZE);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(sharedTime, startEvent, stopEvent);

    cudaEventRecord(startEvent);
    convolutionTexture<<<gridSize, blockSize>>>(texObj, d_outputImageTexture, width, segmentHeightWithOverlap, d_filter, FILTER_SIZE);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(textureTime, startEvent, stopEvent);

    int copyStartRow = (deviceID == 0) ? 0 : overlap;
    int copyEndRow = (deviceID == numGPUs - 1) ? segmentHeightWithOverlap : segmentHeightWithOverlap - overlap;
    int copyHeight = copyEndRow - copyStartRow;
    size_t centralSegmentSize = width * copyHeight * 3 * sizeof(unsigned char);

    cudaMemcpy(h_outputImageShared + yOffset * width * 3, d_outputImageShared + copyStartRow * width * 3, centralSegmentSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outputImageTexture + yOffset * width * 3, d_outputImageTexture + copyStartRow * width * 3, centralSegmentSize, cudaMemcpyDeviceToHost);

    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(d_inputImage);
    cudaFree(d_outputImageShared);
    cudaFree(d_outputImageTexture);
    cudaFree(d_filter);
    free(h_inputImageSegmentRGBA);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

int main() {
    int width, height;
    unsigned char* h_inputImage = readPNG("input.png", &width, &height);
    if (!h_inputImage) {
        printf("Не удалось прочитать входное изображение\n");
        return -1;
    }
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    printf("Number of GPUs available: %d\n", numGPUs);

    float sigma = SIGMA;
    int filterWidth = FILTER_SIZE;
    size_t filterSize = filterWidth * filterWidth * sizeof(float);
    float* h_filter = (float*)malloc(filterSize);
    generateGaussianFilter(h_filter, filterWidth, sigma);

    unsigned char* h_outputImageShared = (unsigned char*)malloc(width * height * 3 * sizeof(unsigned char));
    unsigned char* h_outputImageTexture = (unsigned char*)malloc(width * height * 3 * sizeof(unsigned char));
    float* sharedTimes = (float*)malloc(numGPUs * sizeof(float));
    float* textureTimes = (float*)malloc(numGPUs * sizeof(float));

    for (int i = 0; i < numGPUs; i++) {
        printf("Processing on GPU %d\n", i);
        processOnGPU(i, numGPUs, h_inputImage, h_outputImageShared, h_outputImageTexture, width, height, h_filter, filterWidth, sigma, &sharedTimes[i], &textureTimes[i]);
    }

    writePNG("output_shared.png", width, height, h_outputImageShared);
    writePNG("output_texture.png", width, height, h_outputImageTexture);

    for (int i = 0; i < numGPUs; i++) {
        printf("GPU %d - Shared Memory Kernel Execution Time: %f ms\n", i, sharedTimes[i]);
        printf("GPU %d - Texture Memory Kernel Execution Time: %f ms\n", i, textureTimes[i]);
    }

    free(h_filter);
    free(h_inputImage);
    free(h_outputImageShared);
    free(h_outputImageTexture);
    free(sharedTimes);
    free(textureTimes);

    return 0;
}