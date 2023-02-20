//
// Created by liangfuchu on 23-2-18.
//
#include "yolov5.h"

detect::yolo5::PreprocessorTransform::PreprocessorTransform(const cv::Size &inputSize, const double &f,
                                                            const int &leftWidth, const int &topHeight)
        :_inputSize(inputSize),_f(f),_leftWidth(leftWidth),_topHeight(topHeight)
{
}

cv::Rect detect::yolo5::PreprocessorTransform::transformBbox(const cv::Rect &input) const
{
    cv::Rect r;
    r.x = (input.x - _leftWidth) / _f;
    r.x = MAX(0, MIN(r.x, _inputSize.width - 1));

    r.y = (input.y - _topHeight) / _f;
    r.y = MAX(0, MIN(r.y, _inputSize.width - 1));

    r.width = input.width / _f;
    if (r.x + r.width > _inputSize.width)
    {
        r.width = _inputSize.width - r.x;
    }
    r.height = input.height / _f;
    if (r.y + r.height > _inputSize.height)
    {
        r.height = _inputSize.height - r.y;
    }
    return r;
}

detect::yolo5::Detection::Detection(const int &classId, const cv::Rect &boundingBox, const double &score)
: _classId(classId), _boundingBox(boundingBox), _score(score)
{

}

const int32_t &detect::yolo5::Detection::classId() const noexcept
{
    return _classId;
}

const cv::Rect &detect::yolo5::Detection::boundingBox() const noexcept
{
    return _boundingBox;
}

const double &detect::yolo5::Detection::score() const noexcept
{
    return _score;
}

const std::string &detect::yolo5::Detection::className() const noexcept
{
    return _className;
}

void detect::yolo5::loadEngine(const std::string &filePath, std::unique_ptr<nvinfer1::ICudaEngine> &engine,
                               std::unique_ptr<nvinfer1::IExecutionContext> &context, Logger logger)
{
    std::ifstream file(filePath, std::ios::binary);
    std::vector<char> data;

    file.seekg(0, std::ifstream::end);
    const auto size = file.tellg();
    file.seekg(0, std::ifstream::beg);

    data.resize(size);
    file.read(data.data(), size);
    file.close();

    std::unique_ptr<nvinfer1::IRuntime> trtRuntime(nvinfer1::createInferRuntime(logger));
    engine.reset(trtRuntime->deserializeCudaEngine(data.data(), data.size()));
    context.reset(engine->createExecutionContext());
}

size_t detect::yolo5::getSizeDims(const nvinfer1::Dims &dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

void detect::yolo5::cudaGetMem(std::vector<nvinfer1::Dims> &input_dims, std::vector<nvinfer1::Dims> &output_dims,
                               const std::unique_ptr<nvinfer1::ICudaEngine> &engine, std::vector<void *> &buffers)
{
    // CPU->GPU memory
    for (int i = 0; i < buffers.size(); ++i)
    {
        auto binding_size = getSizeDims(engine->getBindingDimensions(i)) * sizeof(float);

        cudaMalloc(&buffers[i], binding_size);

        if (engine->bindingIsInput(i))
        {
            input_dims.emplace_back(engine->getBindingDimensions(i));
        } else
        {
            output_dims.emplace_back(engine->getBindingDimensions(i));
        }

    }
    // 判断网络是否有误
    if (input_dims.empty() || output_dims.empty())
    {
        std::cerr << "Failed load network" << std::endl;
        exit(1);
    }
}

void detect::yolo5::postprocessResults_0(float *gpu_output, const nvinfer1::Dims &dims,
                                         const PreprocessorTransform &preprocessorTransform,
                                         std::vector<Detection> *out)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classes;

    const int nrClasses = dims.d[2] - 5;
    const int rowsize = dims.d[2];
    const int numGridBoxes = dims.d[1];

    std::vector<float> cpu_output(getSizeDims(dims));
    cudaMemcpyAsync(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    float *begin = cpu_output.data(); //...[]............//...[]..........//

    for (int i = 0; i < numGridBoxes; ++i)
    {
        float *ptr = begin + i * rowsize;

        const float objectness = ptr[4];
        if (objectness < 0.4)
        {
            continue;
        }

        double maxClassScore = 0.0;
        int maxScoreIndex = 0;

        for (int j = 0; j < nrClasses; ++j)
        {
            const float &v = ptr[5 + j];
            if (v > maxClassScore)
            {
                maxClassScore = v;
                maxScoreIndex = j;
            }
        }
        const double score = objectness * maxClassScore;
        if (score < 0.4)
        {
            continue;
        }

        const float w = ptr[2];
        const float h = ptr[3];
        const float x = ptr[0] - w / 2.0;
        const float y = ptr[1] - h / 2.0;

        boxes.emplace_back(cv::Rect(x, y, w, h));
        scores.emplace_back(score);
        classes.emplace_back(maxScoreIndex);
    }

    std::vector<int> indices;

    cv::dnn::NMSBoxes(boxes, scores, 0.4, 0.4, indices);

    for (int j: indices)
    {
        const cv::Rect bbox = preprocessorTransform.transformBbox(boxes[j]);

        const double score = MAX(0.0, MIN(1., scores[j]));

        out->emplace_back(Detection(classes[j], bbox, score));
    }
}

void detect::yolo5::visualizeDetections(cv::Mat &image, std::vector<Detection> &detections)
{
    for (const auto &det: detections)
    {
        /*  bounding box  */
        const cv::Rect &bbox = det.boundingBox();

        std::cout << bbox << "  " << std::endl;

        cv::rectangle(image, bbox, cv::Scalar(255, 0, 0), 2);

        /*  class  */
        std::string className = det.className();
        if (className.length() == 0)
        {
            const int classId = det.classId();
            className = std::to_string(classId);
        }
        cv::putText(image, className,
                    bbox.tl() + cv::Point(0, -10), cv::FONT_HERSHEY_PLAIN,
                    1.0, cv::Scalar(255, 255, 255));

        /*  score */
        const double score = det.score();
        cv::putText(image, std::to_string(score),
                    bbox.tl() + cv::Point(bbox.width, -10),
                    cv::FONT_HERSHEY_PLAIN, 1.0,
                    cv::Scalar(255, 255, 255));
    }
}
