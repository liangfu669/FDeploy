//
// Created by liangfuchu on 23-2-16.
//
#include <iostream>
#include <memory>
#include <vector>

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>


#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include "detect_algorithm/include/yolov5.h"


int main()
{
    std::cout << "Hello, World!" << std::endl;
    std::string image_path = "/home/liangfuchu/code/cpp/vision_cpp/images/bus.jpg";
    std::string filepath="/home/liangfuchu/code/cpp/vision_cpp/weights/test1.engine";

    cv::Mat frame = cv::imread(image_path); //cpu

    // runtime:运行时候的接口实例
    //engine:序列化文件
    //context:管理中间激活的其他状态。

    std::unique_ptr<nvinfer1::IRuntime> trtRuntime(nullptr);
    std::unique_ptr<nvinfer1::ICudaEngine> engine(nullptr);
    std::unique_ptr<nvinfer1::IExecutionContext> context(nullptr);

    detect::yolo5::Logger logge;
    detect::yolo5::loadEngine(filepath, engine, context, logge);

    std::vector<nvinfer1::Dims> input_dims; // we expect only one input
    std::vector<nvinfer1::Dims> output_dims; // and one output
    std::vector<void *> buffers(engine->getNbBindings());

    std::cout<<engine->getNbBindings()<<std::endl;

    detect::yolo5::cudaGetMem(input_dims, output_dims, engine, buffers);

    cv::cuda::GpuMat cuda_frame(frame);
    cv::cuda::Stream _cudaStream;

    const double f = MIN((double) input_dims[0].d[2] / frame.rows,
                         (double) input_dims[0].d[3] / frame.cols);

    const cv::Size boxSize = cv::Size(frame.cols * f, frame.rows * f);

//
    const int dr = input_dims[0].d[2] - boxSize.height;
    const int dc = input_dims[0].d[3] - boxSize.width;
    const int topHeight = std::floor(dr / 2.0);
    const int bottomHeight = std::ceil(dr / 2.0);
    const int leftWidth = std::ceil(dc / 2.0);
    const int rightWidth = std::floor(dc / 2.0);

    cv::cuda::resize(cuda_frame, cuda_frame, boxSize, 0, 0, cv::INTER_LINEAR, _cudaStream);
    cv::cuda::copyMakeBorder(cuda_frame, cuda_frame, topHeight, bottomHeight, leftWidth, rightWidth,
                             cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0), _cudaStream);

    cuda_frame.convertTo(cuda_frame, CV_32FC3, 1.0f / 255.0f, _cudaStream);

    int _networkRows = input_dims[0].d[2];
    int _networkCols = input_dims[0].d[3];
    std::cout << _networkRows << std::endl;
//
    const cv::Size networkSize(_networkCols, _networkRows);

    auto *inputptr = (float *) buffers.at(0);
    std::vector<cv::cuda::GpuMat> channels;


    const int channelSize = networkSize.area();
    cudaMemcpy(inputptr,cuda_frame.data, 3*channelSize* sizeof(float),cudaMemcpyDeviceToDevice);

    //MAt [B] [G] [R]

    //MAT[0][0]  vec3d 11,13,25
    //                  |  |  |
    //MAT[0][1]  vec3d 88 99 100
    //  11 13  25   88 99 100
    //cudaMemcpy  11 13  25   88 99 100 ->buffers [       11 13  25   88 99 100                                                ]

    //11 88 ................  13 99 .................... 25 100...............
    //MAT

    channels.emplace_back(networkSize, CV_32FC1, inputptr + 2 * channelSize);
    /*  G channel will go here  */
    channels.emplace_back(networkSize, CV_32FC1, inputptr + 1 * channelSize);
    /*  R channel will go here  */
    channels.emplace_back(networkSize, CV_32FC1, inputptr);

    cv::cuda::split(cuda_frame, channels, _cudaStream);



    context->enqueueV2(&buffers.front(), nullptr, nullptr);

    detect::yolo5::PreprocessorTransform preprocessorTransform(frame.size(), f, leftWidth, topHeight);//bbox recovery

    std::vector<detect::yolo5::Detection> lst;

    postprocessResults_0((float *) buffers.back(), output_dims.back(), preprocessorTransform, &lst);

    visualizeDetections(frame, lst);



    cv::namedWindow("test");
    cv::imshow("test", frame);
    cv::waitKey(0);


    return 0;
}
