//
// Created by liangfuchu on 23-2-17.
//

#ifndef VISION_CPP_ENHANCEMENT_H
#define VISION_CPP_ENHANCEMENT_H

#include "opencv2/opencv.hpp"


namespace utils {
    namespace Retinex {
        void SSR(const cv::Mat &src, cv::Mat &dst, double sigma);

        void MSR(const cv::Mat &src, cv::Mat &dst, std::vector<double> weights, std::vector<double> sigmas);
    }
    void gamma(const cv::Mat &src, cv::Mat &dst, double gamma);

}


#endif //VISION_CPP_ENHANCEMENT_H
