# PCA---Mini-Project-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD
Mini Project - Face Detection or Convert an image into gray scale image using CUDA GPU programming

## Title
Parallel Image Grayscale Conversion using CUDA GPU Programming

## Abstract
Image processing is one of the most computationally intensive tasks in computer vision. Converting a color image to grayscale is a basic but important preprocessing step in many applications such as object detection, image enhancement, and face recognition.
This project utilizes NVIDIA’s CUDA parallel computing platform to accelerate the grayscale conversion process by running pixel-level computations in parallel on the GPU.
The CUDA-based implementation is compared with a traditional CPU-based method to demonstrate performance improvement achieved through parallel processing.

## Objective
1.Implement a CUDA program to convert an RGB image into grayscale.
2.Utilize parallel GPU threads for pixel-wise computation.
3.Measure and compare performance between CPU and GPU execution.
4.Demonstrate the use of CUDA for basic image processing tasks.

## System Requirements
NVIDIA GPU supporting CUDA
CUDA Toolkit
OpenCV (for reading and writing images)
C++ compiler (nvcc)
Operating System: Windows / Linux

## Methodology

1.Load the Image:
Use OpenCV to read a color image (RGB format).

2.Memory Allocation:
Allocate memory for input (RGB) and output (grayscale) images on both CPU and GPU.

3.CUDA Kernel Design:
Each GPU thread handles one pixel — reading RGB values and computing a single grayscale intensity using:
      Gray=0.299R+0.587G+0.114B


4.Memory Transfer:
Copy the input RGB image data from host (CPU) to device (GPU).

5.Parallel Processing:
Launch CUDA kernel with multiple threads for concurrent pixel processing.

6.Result Retrieval:
Copy the grayscale image back from GPU to CPU and save it using OpenCV.

## Program
```

#include <opencv2/opencv.hpp>
#include <iostream>

_global_ void rgb_to_gray(unsigned char* rgb, unsigned char* gray, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        unsigned char r = rgb[idx];
        unsigned char g = rgb[idx + 1];
        unsigned char b = rgb[idx + 2];
        gray[y * width + x] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

int main() {
    // Load input image
    cv::Mat input = cv::imread("input.jpg", cv::IMREAD_COLOR);
    if (input.empty()) {
        std::cout << "Error: Cannot load image!" << std::endl;
        return -1;
    }

    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();

    cv::Mat output(height, width, CV_8UC1);

    unsigned char *d_rgb, *d_gray;
    size_t rgb_size = width * height * channels * sizeof(unsigned char);
    size_t gray_size = width * height * sizeof(unsigned char);

    cudaMalloc(&d_rgb, rgb_size);
    cudaMalloc(&d_gray, gray_size);

    cudaMemcpy(d_rgb, input.data, rgb_size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    rgb_to_gray<<<grid, block>>>(d_rgb, d_gray, width, height, channels);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data, d_gray, gray_size, cudaMemcpyDeviceToHost);

    cv::imwrite("output_gray.jpg", output);

    std::cout << "Grayscale conversion completed. Output saved as output_gray.jpg\n";

    cudaFree(d_rgb);
    cudaFree(d_gray);

    return 0;
}
```
Use the following commands:


nvcc grayscale.cu -o grayscale pkg-config --cflags --libs opencv4
./grayscale


Input file: input.jpg
Output file: output_gray.jpg

## output
![WhatsApp Image 2025-11-04 at 12 27 23_b6fdf1b4](https://github.com/user-attachments/assets/a1d16125-1d79-49e3-b2bd-1f44c673ce81)

## Result

This project demonstrates the effectiveness of GPU parallel computing using CUDA for image processing tasks.
The grayscale conversion operation, being data-parallel, achieves a significant speedup when executed on the GPU compared to CPU.
This concept can be extended to more complex operations such as edge detection, image filtering, or face detection using CUDA.
