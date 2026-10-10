#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define WIDTH 1920
#define HEIGHT 1080
#define N (WIDTH * HEIGHT)

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", \
                cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

// CUDA Sobel Edge Detection Kernel
__global__ void sobelGPU(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int i = y * width + x;

    // Set border pixels to black
    if (x == 0 || y == 0 ||
        x == width - 1 || y == height - 1) {
        output[i] = 0;
        return;
    }

    // Read neighboring grayscale pixels
    int a = input[(y-1)*width + (x-1)];
    int b = input[(y-1)*width + x];
    int c = input[(y-1)*width + (x+1)];

    int d = input[y*width + (x-1)];
    int f = input[y*width + (x+1)];

    int g = input[(y+1)*width + (x-1)];
    int h = input[(y+1)*width + x];
    int j = input[(y+1)*width + (x+1)];

    // Sobel gradients
    int gx = -a + c - 2*d + 2*f - g + j;
    int gy = -a - 2*b - c + g + 2*h + j;

    float magnitude = sqrtf(
        float(gx*gx + gy*gy)
    );

    // Clamp values to [0, 255]
    output[i] = (unsigned char)fminf(magnitude, 255.0f);
}

int main() {
    size_t bytes = N;

    unsigned char* input =
        (unsigned char*)malloc(bytes);
    unsigned char* output =
        (unsigned char*)malloc(bytes);

    if (!input || !output) {
        fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    // Read previous CUDA grayscale image
    FILE* file = fopen("output.gray", "rb");
    if (!file) {
        printf("Error: output.gray not found\n");
        return 1;
    }

    if (fread(input, 1, bytes, file) != bytes) {
        printf("Error: incomplete grayscale input\n");
        fclose(file);
        return 1;
    }
    fclose(file);

    unsigned char *d_input, *d_output;

    CHECK(cudaMalloc((void**)&d_input, bytes));
    CHECK(cudaMalloc((void**)&d_output, bytes));

    CHECK(cudaMemcpy(d_input, input, bytes,
                     cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks(
        (WIDTH + 15) / 16,
        (HEIGHT + 15) / 16
    );

    // Execute GPU kernel
    sobelGPU<<<blocks, threads>>>(
        d_input, d_output, WIDTH, HEIGHT
    );

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(output, d_output, bytes,
                     cudaMemcpyDeviceToHost));

    FILE* out = fopen("sobel.gray", "wb");
    if (!out) {
        printf("Error: could not save output\n");
        return 1;
    }

    fwrite(output, 1, bytes, out);
    fclose(out);

    printf("CUDA Sobel completed!\n");
    printf("Processed pixels: %d\n", N);
    printf("Threads per block: 256\n");
    printf("Grid dimensions: %d x %d\n",
           (WIDTH + 15)/16, (HEIGHT + 15)/16);

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
    free(input);
    free(output);

    return 0;
}
