//
// Created by liangfuchu on 23-2-17.
//

#ifndef VISION_CPP_ENHANCEMENT_H
#define VISION_CPP_ENHANCEMENT_H

#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"


namespace utils
{
    namespace Retinex
    {
        cv::Mat SSR(cv::Mat src, double sigma);

        cv::Mat MSR(cv::Mat src, std::vector<float> weight, std::vector<float> sigma);

    }

}


#endif //VISION_CPP_ENHANCEMENT_H
