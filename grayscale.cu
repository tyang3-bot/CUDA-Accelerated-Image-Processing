
#include <stdio.h>
#include <cuda_runtime.h>

// GPU Kernel: convert RGB pixels to grayscale
__global__ void grayscaleGPU(
    unsigned char* rgb,
    unsigned char* gray,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int index = i * 3;

        unsigned char r = rgb[index];
        unsigned char g = rgb[index + 1];
        unsigned char b = rgb[index + 2];

        gray[i] = (unsigned char)(
            0.299f * r +
            0.587f * g +
            0.114f * b
        );
    }
}

int main() {
    const int n = 4;

    // Four RGB pixels
    unsigned char rgb[12] = {
        255, 0, 0,      // Red
        0, 255, 0,      // Green
        0, 0, 255,      // Blue
        255, 255, 255   // White
    };

    unsigned char gray[4];

    unsigned char *d_rgb, *d_gray;

    cudaMalloc((void**)&d_rgb, 12);
    cudaMalloc((void**)&d_gray, 4);

    cudaMemcpy(d_rgb, rgb, 12, cudaMemcpyHostToDevice);

    grayscaleGPU<<<1, 4>>>(d_rgb, d_gray, n);

    cudaMemcpy(gray, d_gray, 4, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("Pixel %d: Gray = %d\n", i, gray[i]);
    }

    cudaFree(d_rgb);
    cudaFree(d_gray);

    return 0;
}
