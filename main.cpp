
// -------------- opencv ----------------------- #
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
// ---------------- opencv-cuda ---------------- #
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

// ------------ cuda ------------------------- #
#include <cuda_runtime_api.h>
// ------------------- nvinfer1 ------------------ #
#include "NvInfer.h"

// ------------ standard libraries  --------------- #
#include <iostream>
#include <assert.h>
#include <string>
#include <vector>

// ---------------------------------------------- #

void preprocessImage(const std::string &image_path, float *gpu_input,
                     nvinfer1::Dims3 &dims) {
    // read image
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty()) {
        std::cerr << "failed to load image: " << image_path << "!" << std::endl;
        return;
    }
    // upload
    cv::cuda::GpuMat gpu_frame;
    gpu_frame.upload(frame);

    // resize
    // CHW order
    auto input_width = dims.d[2];
    auto input_height = dims.d[1];
    auto channels = dims.d[0];

    auto input_size = cv::Size(input_width, input_height);
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_LINEAR);

    //*  ------------------------ Pytorch ToTensor and Normalize ------------------- */
    cv::cuda::GpuMat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);

    cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.346f, 0.406f), flt_image,
                       cv::noArray(), -1);

    cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);
    //* ----------------------------------------------------------------------------------- /
    // BGR To RGB
    cv::cuda::GpuMat rgb;
    cv::cuda::cvtColor(flt_image, rgb, cv::COLOR_BGR2RGB);

    // toTensor(copy data to input float pointer channel by channel)
    std::vector<cv::cuda::GpuMat> rgb_out;
    for (size_t i = 0; i < channels; ++i) {
        rgb_out.emplace_back(cv::cuda::GpuMat(cv::Size(input_width, input_height), CV_32FC1,
                                              gpu_input + i * input_width * input_height));
    }

    cv::cuda::split(flt_image, rgb_out); // opencv HWC order -> CHW order
}

// calculate size of tensor
size_t getSizeByDim(const nvinfer1::Dims &dims) {
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }
    return size;
}

int main() {
    std::string image_path = "../00.jpg";
    // CHW order
    nvinfer1::Dims3 input_dim(3, 448, 448);

    auto input_size = getSizeByDim(input_dim) * sizeof(float);
    // allocate gpu memory for network inference
    // 此处的buffer可以认为是TensorRT engine推理时在GPU上分配的输入显存
    std::vector<void *> buffers(1);
    cudaMalloc(&buffers[0], input_size);

    // preprocess
    preprocessImage(image_path, (float *) buffers[0], input_dim);

    // download
    cv::cuda::GpuMat gpu_output;
    std::vector<cv::cuda::GpuMat> resized;
    for (size_t i = 0; i < 3; ++i) {
        resized.emplace_back(cv::cuda::GpuMat(cv::Size(input_dim.d[2], input_dim.d[1]), CV_32FC1,
                                              (float *) buffers[0] + i * input_dim.d[2] * input_dim.d[1]));
    }
    cv::cuda::merge(resized, gpu_output);

    cv::cuda::GpuMat image_out;
    // normalize
    gpu_output.convertTo(image_out, CV_32FC3, 1.f * 255.f);
    // download
    cv::Mat dst;
    image_out.download(dst);

    cv::imwrite("../01_test_demo.jpg", dst);

    for (void *buf: buffers) {
        cudaFree(buf);
    }

    return 0;
}