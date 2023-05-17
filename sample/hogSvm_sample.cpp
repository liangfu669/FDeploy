//
// Created by liangfuchu on 23-3-1.
//
#include "../algorithm/classify/include/hogSvm.h"


int main()
{
    std::string svmPath = "/home/liangfuchu/code/cpp/vision_cpp/train/hogSvm/svm_model.xml";
    std::string hogPath = "/home/liangfuchu/code/cpp/vision_cpp/train/hogSvm/hogParm.yaml";
    std::string imagePath = "/home/liangfuchu/code/cpp/vision_cpp/images/6.png";
    cv::Mat image = cv::imread(imagePath);

    auto detector = classify::HogSvmDetect(svmPath, hogPath);

    float res = detector.infer(image);
    std::cout << res << std::endl;
    return 0;
}