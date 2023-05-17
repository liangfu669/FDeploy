//
// Created by liangfuchu on 23-2-25.
//

#ifndef VISION_CPP_HOGSVM_H
#define VISION_CPP_HOGSVM_H

#include "opencv2/opencv.hpp"

namespace classify
{
    class HogSvmDetect
    {
    public:
        cv::HOGDescriptor hog;
        cv::Ptr<cv::ml::SVM> svm;
        explicit HogSvmDetect(const std::string& svmPath, const std::string& hogPath);
        float infer(const cv::Mat& image) const;
    };
}

#endif //VISION_CPP_HOGSVM_H
