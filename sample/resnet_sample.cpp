#include "opencv2/opencv.hpp"
#include "../classification_algorithm/include//resnet.h"

int main() {
    std::string path = "/home/liangfuchu/code/cpp/vision_cpp/weights/resnet18.engine";
    cv::Mat image = cv::imread("/home/liangfuchu/code/cpp/vision_cpp/images/bobby.jpg");

    classify::Resnet resnet(path, "/home/liangfuchu/code/cpp/vision_cpp/classification_algorithm/imagenet_classes.txt",
                            224, 224, 1000);
    auto result = resnet.infer(image);
    auto class_id = std::get<0>(result);
    auto class_name = std::get<2>(result);
    auto score = std::get<1>(result);

    std::cout << "class_id: " << class_id << std::endl;
    std::cout << "class_name: " << class_name << std::endl;
    std::cout << "score: " << score << std::endl;
    std::cout << resnet.engine->getNbBindings() << std::endl;
//    cv::imshow("image", image);
//    cv::waitKey(0);
    return 0;
}