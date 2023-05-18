#include <iostream>
#include "../algorithm/segment/include/unet.hpp"
#include "opencv2/opencv.hpp"

int main() {
    auto model = unet::load("weights/unet.engine",
                            unet::Type::UNET);

    if (model == nullptr) {
        std::cout << "fail load\n";
    }
    cv::Mat img = cv::imread("image/street.jpg");
    auto res = model->forward(img);
    unet::render(img, std::get<0>(res), std::get<1>(res));
    cv::imshow("seg_img", img);
    cv::waitKey(0);
    cv::imwrite("image/seg_street.jpg", img);
    return 0;
}

