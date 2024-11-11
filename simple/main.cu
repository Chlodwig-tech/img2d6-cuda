#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstring> 

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../stb/stb_image.h"
#include "../stb/stb_image_write.h"

#define DICESIZE 10

#define CUDA_CALL(x, message) {if((x) != cudaSuccess) { \
    printf("Error - %s(%d)[%s]: %s\n", __FILE__, __LINE__, message, cudaGetErrorString(x)); \
    exit(EXIT_FAILURE); }}

#define DICE(dimg, idx, cond) \
    (dimg[idx] = dimg[idx + 1] = dimg[idx + 2] = (cond) ? 0 : 255)

__global__ void img2d6Kernel(unsigned char *dimg, int width, int height, int channels){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ unsigned int avg[DICESIZE - 1][DICESIZE - 1];

    if(row >= height || col >= width)
        return;

    int idx = (row * width + col) * channels;
    if(threadIdx.x == DICESIZE - 1 || threadIdx.y == DICESIZE - 1){
        dimg[idx] = dimg[idx + 1] = dimg[idx + 2] = 0;
        return;
    }

    unsigned char grey = (unsigned char)(0.299f * dimg[idx] + 0.587f * dimg[idx + 1] + 0.114f * dimg[idx + 2]);
    int x = threadIdx.x, y = threadIdx.y;
    avg[x][y] = grey;
    __syncthreads();
    if(x == 0){
        for(int i = 1; i < DICESIZE - 1; i++){
            avg[0][y] += avg[i][y];
        }   
    }
    __syncthreads();
    if(x == 0 && y == 0){
        for(int i = 1; i < DICESIZE - 1; i++){
            avg[0][0] += avg[0][i];
        }
        int d = avg[0][0] / ((DICESIZE - 1) * (DICESIZE - 1)) *6 / 255;
        avg[0][0] = d > 5 ? 6 : d + 1;;
    }
    __syncthreads();

    int d = 7 - avg[0][0];
    switch(d){
        case 1:
            DICE(dimg, idx, x >= 3 && x <= 5 && y >= 3 & y <= 5);
            break;
        case 2:
            DICE(dimg, idx, (x >= 6 && y < 3) || (x < 3 && y >= 6));
            break;
        case 3:
            DICE(dimg, idx, (x >= 3 && x <= 5 && y >= 3 & y <= 5) || (x >= 6 && y < 3) || (x < 3 && y >= 6));
            break;
        case 4:
            DICE(dimg, idx, (x >= 6 && y < 3) || (x < 3 && y >= 6) || (x < 3 && y < 3) || (x >= 6 && y >= 6));
            break;
        case 5:
            DICE(dimg, idx, (x >= 6 && y < 3) || (x < 3 && y >= 6) || (x < 3 && y < 3) || (x >= 6 && y >= 6) || (x >=3 && x <= 5 && y >= 3 & y <= 5));
            break;
        case 6:
            DICE(dimg, idx, y < 3 || y >= 6);
            break;
    }
}


int main(int argc, char** argv) {

    if (argc != 2) {
        printf("Usage: %s <image_path>\n", argv[0]);
        return -1;
    }
    const char* input_img = argv[1];
    int len = strlen(input_img);
    char* output_img = new char[len + 7];
    strcpy(output_img, "output_");
    strcat(output_img, input_img);

    int width, height, channels;
    unsigned char *himg = stbi_load(input_img, &width, &height, &channels, 3);
    unsigned char *dimg;
    int size = width * height * channels * sizeof(unsigned char);

    CUDA_CALL(cudaMalloc((void **)&dimg, size), "cudaMalloc - dimg");

    CUDA_CALL(cudaMemcpy(dimg, himg, size, cudaMemcpyHostToDevice), "cudaMemcpy - himg -> dimg");

    int nn = DICESIZE;
    dim3 block_size(nn, nn);
    dim3 grid_size(
        (height - 1) / block_size.x + 1,
        (width - 1) / block_size.y + 1
    );

    img2d6Kernel<<<grid_size, block_size>>>(dimg, width, height, channels);

    CUDA_CALL(cudaMemcpy(himg, dimg, size, cudaMemcpyDeviceToHost), "cudaMemcpy - dimg -> himg");

    stbi_write_png(output_img, width, height, channels, himg, width * channels);

    CUDA_CALL(cudaFree(dimg), "cudaFree - dimg")
    printf("Saved to %s\n", output_img);

    delete output_img;

    return 0;
}
