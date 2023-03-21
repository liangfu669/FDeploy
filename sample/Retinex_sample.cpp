//
// Created by liangfuchu on 23-3-20.
//
#include "../utils_algorithm/include/enhancement.h"
#include "opencv2/opencv.hpp"

int main() {
    cv::Mat img = cv::imread("../../images/4_2-1.png");
    cv::Mat dst;
    utils::Retinex::SSR(img, dst, 300);
    cv::imshow("img", dst);
    cv::waitKey(0);
}
