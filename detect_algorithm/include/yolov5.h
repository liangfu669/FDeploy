//
// Created by liangfuchu on 23-2-18.
//

#ifndef VISION_CPP_YOLOV5_H
#define VISION_CPP_YOLOV5_H

#include <memory>
#include <fstream>

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>


namespace detect
{
    namespace yolo5
    {
        class Logger : public nvinfer1::ILogger
        {
        public:
            void log(Severity severity, char const *msg) noexcept override
            {
            }
        };


        class PreprocessorTransform
        {
        public:
            PreprocessorTransform(const cv::Size &inputSize, const double &f, const int &leftWidth,
                                  const int &topHeight);

        public:
            cv::Rect transformBbox(const cv::Rect &input) const;

        private:
            cv::Size _inputSize;
            double _f;
            int _leftWidth;
            int _topHeight;
        };


        struct Result
        {
            const int32_t classId;
            const cv::Rect boundingBox;
            double score;
            std::string className;

            Result(int32_t classId, cv::Rect boundingBox, double score);
        };


        void loadEngine(const std::string &filePath, std::unique_ptr<nvinfer1::ICudaEngine> &engine,
                        std::unique_ptr<nvinfer1::IExecutionContext> &context, Logger logger);


        size_t getSizeDims(const nvinfer1::Dims &dims);


        void cudaGetMem(std::vector<nvinfer1::Dims> &input_dims, std::vector<nvinfer1::Dims> &output_dims,
                        const std::unique_ptr<nvinfer1::ICudaEngine> &engine, std::vector<void *> &buffers);


        void postprocessResults_0(float *gpu_output, const nvinfer1::Dims &dims,
                                  const PreprocessorTransform &preprocessorTransform, std::vector<Result> *out);


        void visualizeDetections(cv::Mat &image, std::vector<Result> &results);

        class Detector
        {
        public:
            Detector(const std::string& weight_path, Logger logger);
            std::vector<Result> infer(cv::Mat image, std::vector<Result> &results);
        private:
            std::unique_ptr<nvinfer1::ICudaEngine> engine;
            std::unique_ptr<nvinfer1::IExecutionContext> context;
        };
    }
}
#endif //VISION_CPP_YOLOV5_H
