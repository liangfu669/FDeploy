#ifndef VISION_CPP_RESNET_HPP
#define VISION_CPP_RESNET_HPP

#include "opencv2/opencv.hpp"
#include "fstream"

namespace resnet {

    typedef std::tuple<int, float> Result;

    class Infer {
    public:
        virtual Result forward(const cv::Mat &image, void *stream = nullptr) = 0;

        virtual std::vector<Result> forwards(const std::vector<cv::Mat> &images,
                                             void *stream = nullptr) = 0;
    };

    std::shared_ptr<Infer> load(const std::string &engine_file);

    std::vector<std::string> read_class_name(const std::string &path);


};
#endif //VISION_CPP_RESNET_HPP
