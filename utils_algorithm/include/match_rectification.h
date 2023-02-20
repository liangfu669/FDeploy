//
// Created by liangfuchu on 23-2-16.
//

#ifndef VISION_CPP_MATCH_RECTIFICATION_H
#define VISION_CPP_MATCH_RECTIFICATION_H

#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"


namespace utils
{
    cv::Mat match(const cv::Mat& image1, const cv::Mat& image2);

    cv::Mat rectification(const cv::Mat& image, cv::Mat H, cv::Size Size);
}

#endif //VISION_CPP_MATCH_RECTIFICATION_H
