
#include "../include/hogSvm.h"

classify::HogSvmDetect::HogSvmDetect(const std::string& svmPath, const std::string& hogPath)
{
    svm = cv::ml::SVM::load(svmPath);
    hog.load(hogPath);
}

float classify::HogSvmDetect::infer(const cv::Mat& image) const
{
    cv::Mat image_re;
    cv::resize(image, image_re, hog.winSize);
    std::vector<float> hogDes;
    hog.compute(image_re, hogDes);
    cv::Mat hogMatDes = cv::Mat::zeros(cv::Size(hogDes.size(), 1), CV_32FC1);
    for (int i = 0; i < hogDes.size(); i++)
        hogMatDes.at<float>(0, i) = hogDes[i];
    float result = svm->predict(hogMatDes);
    return result;
}

