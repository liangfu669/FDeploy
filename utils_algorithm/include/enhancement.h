//
// Created by liangfuchu on 23-2-17.
//

#ifndef VISION_CPP_ENHANCEMENT_H
#define VISION_CPP_ENHANCEMENT_H

#include "Eigen/Dense"
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"
#include "cmath"


namespace utils {
    namespace Retinex {
        void SSR(cv::Mat src, cv::Mat &dst, double sigma);

        void MSR(cv::Mat src, cv::Mat &dst, std::vector<float> weight, std::vector<float> sigma);

    }

}


#endif //VISION_CPP_ENHANCEMENT_H
