#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define WIDTH 1920
#define HEIGHT 1080
#define N (WIDTH * HEIGHT)

// One GPU thread processes one pixel
__global__ void grayscaleGPU(
    const unsigned char* rgb,
    unsigned char* gray,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int j = i * 3;
        gray[i] = (unsigned char)(
            0.299f * rgb[j] +
            0.587f * rgb[j + 1] +
            0.114f * rgb[j + 2]
        );
    }
}

int main() {
    size_t rgbBytes = N * 3;
    size_t grayBytes = N;

    unsigned char* rgb = (unsigned char*)malloc(rgbBytes);
    unsigned char* gray = (unsigned char*)malloc(grayBytes);

    FILE* file = fopen("input.rgb", "rb");
    if (!file) {
        printf("Error: input.rgb not found\n");
        return 1;
    }

    if (fread(rgb, 1, rgbBytes, file) != rgbBytes) {
        printf("Error: incomplete input file\n");
        fclose(file);
        return 1;
    }
    fclose(file);

    unsigned char *d_rgb, *d_gray;
    cudaMalloc((void**)&d_rgb, rgbBytes);
    cudaMalloc((void**)&d_gray, grayBytes);

    // CPU -> GPU
    cudaMemcpy(d_rgb, rgb, rgbBytes,
               cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Execute CUDA kernel
    grayscaleGPU<<<blocks, threads>>>(d_rgb, d_gray, N);
    cudaDeviceSynchronize();

    // GPU -> CPU
    cudaMemcpy(gray, d_gray, grayBytes,
               cudaMemcpyDeviceToHost);

    FILE* out = fopen("output.gray", "wb");
    if (!out) {
        printf("Error: cannot create output.gray\n");
        return 1;
    }
    fwrite(gray, 1, grayBytes, out);
    fclose(out);

    printf("Processed %d pixels using CUDA!\n", N);
    printf("Threads per block: %d\n", threads);
    printf("Number of blocks: %d\n", blocks);

    cudaFree(d_rgb);
    cudaFree(d_gray);
    free(rgb);
    free(gray);
    return 0;
}
