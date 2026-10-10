#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define N (1920 * 1080)

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", \
                cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

// CPU version
void grayscaleCPU(const unsigned char* rgb,
                  unsigned char* gray) {
    for (int i = 0; i < N; i++) {
        int j = i * 3;
        gray[i] = (unsigned char)(
            0.299f * rgb[j] +
            0.587f * rgb[j + 1] +
            0.114f * rgb[j + 2]);
    }
}

// GPU version
__global__ void grayscaleGPU(const unsigned char* rgb,
                             unsigned char* gray) {
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

    unsigned char* rgb = (unsigned char*)malloc(rgbBytes);
    unsigned char* cpu = (unsigned char*)malloc(grayBytes);
    unsigned char* gpu = (unsigned char*)malloc(grayBytes);

    FILE* f = fopen("input.rgb", "rb");
    if (!f || fread(rgb, 1, rgbBytes, f) != rgbBytes) {
        printf("Could not read input.rgb\n");
        return 1;
    }
    fclose(f);

    unsigned char *d_rgb, *d_gray;
    CHECK(cudaMalloc(&d_rgb, rgbBytes));
    CHECK(cudaMalloc(&d_gray, grayBytes));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    const int reps = 100;

    // CPU warm-up
    grayscaleCPU(rgb, cpu);

    auto c1 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++)
        grayscaleCPU(rgb, cpu);
    auto c2 = std::chrono::steady_clock::now();

    double cpuMs =
        std::chrono::duration<double, std::milli>(
            c2 - c1).count() / reps;

    CHECK(cudaMemcpy(d_rgb, rgb, rgbBytes,
                     cudaMemcpyHostToDevice));

    // GPU warm-up
    grayscaleGPU<<<blocks, threads>>>(d_rgb, d_gray);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Kernel-only timing
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    for (int r = 0; r < reps; r++)
        grayscaleGPU<<<blocks, threads>>>(d_rgb, d_gray);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0;
    CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
    double gpuMs = elapsedMs / reps;

    CHECK(cudaMemcpy(gpu, d_gray, grayBytes,
                     cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        if (abs((int)cpu[i] - (int)gpu[i]) > 1)
            mismatches++;
    }

    printf("CPU average: %.4f ms\n", cpuMs);
    printf("GPU kernel average: %.4f ms\n", gpuMs);
    printf("Kernel-only speedup: %.2fx\n", cpuMs / gpuMs);
    printf("Mismatched pixels (>1): %d\n", mismatches);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_rgb));
    CHECK(cudaFree(d_gray));
    free(rgb);
    free(cpu);
    free(gpu);
    return 0;
}
