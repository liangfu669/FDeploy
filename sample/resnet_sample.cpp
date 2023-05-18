#include <iostream>
#include "../algorithm/classify/include/resnet.hpp"
#include "opencv2/opencv.hpp"

int main() {
    auto model = resnet::load("weights/resnet.engine");
    auto img = cv::imread("image/ng.jpg");
    auto res = model->forward(img);
    auto class_name = resnet::read_class_name(
            "files/labels.imagenet.txt");
    std::cout << class_name[std::get<0>(res)] << " " << std::get<1>(res);
    return 0;
}
