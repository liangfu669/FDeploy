//
// Created by liangfuchu on 23-3-2.
//

#include "../include/resnet.h"

classify::Resnet::Resnet(const std::string &enginePath, const std::string &classNamePath, const int &inputH,
                         const int &inputW, const int &outputSize) {
    std::ifstream file(enginePath, std::ios::binary);
    size_t size;
    char *trtModelStream;
    if (file.good()) {
        file.seekg(0, std::ifstream::end);
        size = file.tellg();
        file.seekg(0, std::ifstream::beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    assert(runtime != nullptr);
    engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(trtModelStream, size));
    assert(engine != nullptr);
    context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    assert(context != nullptr);
    delete[] trtModelStream;

    class_names = classify::read_class_name(classNamePath);
    _inputH = inputH;
    _inputW = inputW;
    _outputSize = outputSize;
}

std::tuple<int, float, std::string> classify::Resnet::infer(const cv::Mat &image) {
    cv::cuda::GpuMat gpuMat(image);
    cv::cuda::resize(gpuMat, gpuMat, cv::Size(_inputW, _inputH));
    cv::cuda::cvtColor(gpuMat, gpuMat, cv::COLOR_BGR2RGB);
    gpuMat.convertTo(gpuMat, CV_32FC3, 1.0 / 255.0);

    void *buffers[2];
    cudaMalloc(&buffers[0], _inputH * _inputW * 3 * sizeof(float));
    std::vector<cv::cuda::GpuMat> gpuRGB;
    for (int i = 0; i < 3; ++i) {
        gpuRGB.emplace_back(cv::Size(_inputW, _inputH), CV_32FC1,
                            (float *) buffers[0] + i * _inputH * _inputW);
    }
    cv::cuda::split(gpuMat, gpuRGB);

    cudaMalloc(&buffers[1], _outputSize * sizeof(float));

    context->execute(1, buffers);

    auto *output = new float[_outputSize];
    cudaMemcpy(output, buffers[1], _outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    std::tuple<int, float, std::string> result{0, 0.0, ""};

    for (int i = 0; i < _outputSize; ++i) {
        if (std::get<1>(result) < output[i]) {
            result = std::make_tuple(i, output[i], class_names[i]);
        }
    }

    return result;
}

