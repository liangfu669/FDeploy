
#include <opencv2/opencv.hpp>

#include "algorithm/common/include/cpm.hpp"
#include "algorithm/detect/include/yolo.hpp"
#include "algorithm/classify/include/resnet.hpp"
#include "algorithm/segment/include/unet.hpp"
#include "algorithm/common/include/infer.hpp"

using namespace std;


yolo::Image cvimg(const cv::Mat &image) { return {image.data, image.cols, image.rows}; }

void perf() {
    int max_infer_batch = 16;
    int batch = 16;
    std::vector<cv::Mat> images{cv::imread("image/car.jpg"), cv::imread("image/gril.jpg"),
                                cv::imread("image/group.jpg")};

    for (int i = images.size(); i < batch; ++i) images.push_back(images[i % 3]);

    cpm::Instance<yolo::BoxArray, yolo::Image, yolo::Infer> cpmi;
    bool ok = cpmi.start([] { return yolo::load("weights/yolov8n.transd.engine", yolo::Type::V8); },
                         max_infer_batch);

    if (!ok) return;

    std::vector<yolo::Image> yoloimages(images.size());
    std::transform(images.begin(), images.end(), yoloimages.begin(), cvimg);

    trt::Timer timer;
    for (int i = 0; i < 5; ++i) {
        timer.start();
        cpmi.commits(yoloimages).back().get();
        timer.stop("BATCH16");
    }

    for (int i = 0; i < 5; ++i) {
        timer.start();
        cpmi.commit(yoloimages[0]).get();
        timer.stop("BATCH1");
    }
}


int main() {
    perf();

    return 0;
}