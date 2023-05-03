//
// Created by liangfuchu on 23-3-2.
//

#ifndef VISION_CPP_RESNET_H
#define VISION_CPP_RESNET_H

#include <memory>
#include <fstream>

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include "thrust/device_vector.h"

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"

namespace classify {
    using namespace nvinfer1;


    std::vector<std::string> read_class_name(const std::string &path) {
        std::vector<std::string> class_names;
        std::ifstream infile(path);
        std::string line;
        while (std::getline(infile, line)) {
            class_names.push_back(line);
        }
        return class_names;
    }

    class Logger : public ILogger {
        void log(Severity severity, const char *msg) noexcept override {
            // suppress info-level messages
            if (severity <= Severity::kWARNING)
                std::cout << msg << std::endl;
        }
    };

    class Resnet {
        public:
        Resnet(const std::string &enginePath, const std::string &classNamePath, const int &inputH, const int &inputW,
               const int &outputSize);

        std::tuple<int, float, std::string> infer(const cv::Mat &image);

        Logger logger;

        std::unique_ptr<ICudaEngine> engine;
        std::unique_ptr<IExecutionContext> context;
        std::vector<std::string> class_names;

        int _inputH;
        int _inputW;
        int _outputSize;
    };
}

#endif //VISION_CPP_RESNET_H
