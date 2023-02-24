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

    std::vector<detect::yolo5::Result> results;
    detect::yolo5::Logger logger;
    detect::yolo5::Detector detector(filepath, logger);
    detector.infer(frame, results);

    visualizeDetections(frame, results);



    cv::namedWindow("test");
    cv::imshow("test", frame);
    cv::waitKey(0);


    return 0;
}
