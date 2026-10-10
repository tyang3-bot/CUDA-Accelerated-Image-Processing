#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define N (1920 * 1080)

__global__ void grayscaleGPU(
    const unsigned char* rgb,
    unsigned char* gray
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int j = i * 3;
        gray[i] = (unsigned char)(
            0.299f * rgb[j] +
            0.587f * rgb[j + 1] +
            0.114f * rgb[j + 2]);
    }
}

int main() {
    size_t rgbBytes = N * 3;
    size_t grayBytes = N;

    unsigned char *rgb = (unsigned char*)malloc(rgbBytes);
    unsigned char *gray = (unsigned char*)malloc(grayBytes);

    FILE* f = fopen("input.rgb", "rb");
    if (!f) return 1;
    if (fread(rgb, 1, rgbBytes, f) != rgbBytes) {
        fclose(f);
        return 1;
    }
    fclose(f);

    unsigned char *d_rgb, *d_gray;
    cudaMalloc(&d_rgb, rgbBytes);
    cudaMalloc(&d_gray, grayBytes);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Warm up GPU
    cudaMemcpy(d_rgb, rgb, rgbBytes,
               cudaMemcpyHostToDevice);
    grayscaleGPU<<<blocks, threads>>>(d_rgb, d_gray);
    cudaDeviceSynchronize();

    double total = 0;
    const int reps = 100;

    for (int r = 0; r < reps; r++) {
        auto start = std::chrono::steady_clock::now();

        cudaMemcpy(d_rgb, rgb, rgbBytes,
                   cudaMemcpyHostToDevice);

        grayscaleGPU<<<blocks, threads>>>(d_rgb, d_gray);

        cudaMemcpy(gray, d_gray, grayBytes,
                   cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();

        auto stop = std::chrono::steady_clock::now();

        total += std::chrono::duration<double,
                 std::milli>(stop - start).count();
    }

    printf("GPU end-to-end average: %.4f ms\n",
           total / reps);

    printf("CPU average (previous test): 4.8806 ms\n");
    printf("End-to-end speedup: %.2fx\n",
           4.8806 / (total / reps));

    cudaFree(d_rgb);
    cudaFree(d_gray);
    free(rgb);
    free(gray);
    return 0;
}
