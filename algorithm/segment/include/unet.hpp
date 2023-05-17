#ifndef __UNET_HPP__
#define __UNET_HPP__

#include <future>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace unet {

    enum class Type : int {
        UNET = 1,
        U2NET = 2
    };

    typedef std::tuple<cv::Mat, cv::Mat> Result;

    class Infer {
    public:
        virtual Result forward(const cv::Mat &image, void *stream = nullptr) = 0;

        virtual std::vector<Result> forwards(const std::vector<cv::Mat> &images,
                                               void *stream = nullptr) = 0;
    };

    std::shared_ptr<Infer> load(const std::string &engine_file, Type type);


    void render(cv::Mat &image, const cv::Mat &prob, const cv::Mat &iclass);


};

#endif