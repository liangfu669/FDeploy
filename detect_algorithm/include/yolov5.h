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
            virtual void log(Severity severity, char const *msg) noexcept override
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


        class Detection
        {
        public:
            Detection(const int &classId, const cv::Rect &boundingBox, const double &score);

            const int32_t &classId() const noexcept;


            const cv::Rect &boundingBox() const noexcept;


            const double &score() const noexcept;


            const std::string &className() const noexcept;

        private:
            int32_t _classId;
            std::string _className;

            cv::Rect _boundingBox;
            double _score;
        };


        void loadEngine(const std::string &filePath, std::unique_ptr<nvinfer1::ICudaEngine> &engine,
                        std::unique_ptr<nvinfer1::IExecutionContext> &context, Logger logger);


        size_t getSizeDims(const nvinfer1::Dims &dims);


        void cudaGetMem(std::vector<nvinfer1::Dims> &input_dims, std::vector<nvinfer1::Dims> &output_dims,
                        const std::unique_ptr<nvinfer1::ICudaEngine> &engine, std::vector<void *> &buffers);


        void postprocessResults_0(float *gpu_output, const nvinfer1::Dims &dims,
                                  const PreprocessorTransform &preprocessorTransform, std::vector<Detection> *out);


        void visualizeDetections(cv::Mat &image, std::vector<Detection> &detections);
    }
}
#endif //VISION_CPP_YOLOV5_H
