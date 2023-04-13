//
// Created by liangfuchu on 23-3-20.
//
#include "../utils_algorithm/include/enhancement.h"
#include "opencv2/opencv.hpp"

using namespace utils;

int main() {
    cv::Mat img = cv::imread("/home/liangfuchu/code/cpp/vision_cpp/images/img.png");
    cv::Mat dst;
    Retinex::SSR(img, dst, 300);
    cv::imshow("ssr", dst);

    cv::Mat msr;
    std::vector<double> weights{0.33, 0.33, 0.34};
    std::vector<double> sigmas{40, 100, 300};
    Retinex::MSR(img, msr, weights, sigmas);
    cv::imshow("msr", msr);

    cv::Mat gma;
    double g = 0.3;
    gamma(img, gma, g);
    cv::imshow("gamma", gma);
    cv::waitKey(0);
}
