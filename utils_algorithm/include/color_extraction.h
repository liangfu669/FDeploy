//
// Created by liangfuchu on 23-2-16.
//

#ifndef VISION_CPP_COLOR_EXTRACTION_H
#define VISION_CPP_COLOR_EXTRACTION_H

#include "opencv2/opencv.hpp"

namespace utils
{
    class HsvExtraction
    {
    public:
        std::pair<cv::Scalar, cv::Scalar> black = {cv::Scalar(0, 0, 0), cv::Scalar(180, 255, 46)};
        std::pair<cv::Scalar, cv::Scalar> grey = {cv::Scalar(0, 0, 46), cv::Scalar(180, 43, 220)};
        std::pair<cv::Scalar, cv::Scalar> white = {cv::Scalar(0, 0, 221), cv::Scalar(180, 30, 255)};
        std::pair<cv::Scalar, cv::Scalar> orange = {cv::Scalar(11, 43, 46), cv::Scalar(25, 255, 255)};
        std::pair<cv::Scalar, cv::Scalar> yellow = {cv::Scalar(26, 43, 46), cv::Scalar(34, 255, 255)};
        std::pair<cv::Scalar, cv::Scalar> green = {cv::Scalar(35, 43, 46), cv::Scalar(77, 255, 255)};
        std::pair<cv::Scalar, cv::Scalar> cyan = {cv::Scalar(78, 43, 46), cv::Scalar(99, 255, 255)};
        std::pair<cv::Scalar, cv::Scalar> blue = {cv::Scalar(100, 43, 46), cv::Scalar(124, 255, 255)};
        std::pair<cv::Scalar, cv::Scalar> purple = {cv::Scalar(125, 43, 46), cv::Scalar(155, 255, 255)};
        std::pair<cv::Scalar, cv::Scalar> red1 = {cv::Scalar(0, 43, 46), cv::Scalar(10, 255, 255)};
        std::pair<cv::Scalar, cv::Scalar> red2 = {cv::Scalar(156, 43, 46), cv::Scalar(180, 255, 255)};

        cv::Mat extraction_color(cv::Mat image, const std::string &color) const;
    };
}

#endif //VISION_CPP_COLOR_EXTRACTION_H
