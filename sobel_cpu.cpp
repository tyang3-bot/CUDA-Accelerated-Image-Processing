#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define WIDTH 1920
#define HEIGHT 1080
#define N (WIDTH * HEIGHT)

void sobelCPU(const unsigned char* input,
              unsigned char* output) {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int i = y * WIDTH + x;

            if (x == 0 || y == 0 ||
                x == WIDTH-1 || y == HEIGHT-1) {
                output[i] = 0;
                continue;
            }

            int a = input[(y-1)*WIDTH + x-1];
            int b = input[(y-1)*WIDTH + x];
            int c = input[(y-1)*WIDTH + x+1];
            int d = input[y*WIDTH + x-1];
            int f = input[y*WIDTH + x+1];
            int g = input[(y+1)*WIDTH + x-1];
            int h = input[(y+1)*WIDTH + x];
            int j = input[(y+1)*WIDTH + x+1];

            int gx = -a+c-2*d+2*f-g+j;
            int gy = -a-2*b-c+g+2*h+j;

            float magnitude = sqrtf(float(gx*gx+gy*gy));
            output[i] = (unsigned char)fminf(magnitude,255.0f);
        }
    }
}

int main() {
    unsigned char* input = (unsigned char*)malloc(N);
    unsigned char* output = (unsigned char*)malloc(N);

    FILE* f = fopen("output.gray", "rb");
    if (!f || fread(input,1,N,f) != N) {
        printf("Input error\n");
        return 1;
    }
    fclose(f);

    sobelCPU(input,output); // Warm-up

    const int reps = 100;
    auto start = std::chrono::steady_clock::now();

    for (int r=0;r<reps;r++)
        sobelCPU(input,output);

    auto stop = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double,std::milli>(
        stop-start).count()/reps;

    FILE* out = fopen("sobel_cpu.gray","wb");
    if (!out) return 1;
    fwrite(output,1,N,out);
    fclose(out);

    printf("CPU Sobel average: %.4f ms\n",ms);
    free(input);
    free(output);
}
