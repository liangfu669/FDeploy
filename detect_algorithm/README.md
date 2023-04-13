## detect_algorithm
检测算法已用tenosrrt部署了yolov5,使用教程：
```shell
git clone https://github.com/ultralytics/yolov5
python export.py --weights "你训练好的yolov5权重" --include engine
```
### opencv_cuda 预处理，cpu后处理解码示例
将detect_algorithm文件下include/yolov5.h src/yolov5.cpp编译成动态链接库。
该库提供最简易的yolov5部署使用方式，使用事例如下：
```c++
#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include "../detect_algorithm/include/yolov5.h"


int main()
{
    std::string image_path = "/home/liangfuchu/code/cpp/vision_cpp/images/bus.jpg";
    std::string filepath="/home/liangfuchu/code/cpp/vision_cpp/weights/test1.engine";

    cv::Mat frame = cv::imread(image_path); //cpu

    std::vector<detect::yolo5::Result> results;
    detect::yolo5::Detector detector(filepath);
    detector.infer(frame, results);     //得到物体的位置信息，置信度信息

    visualizeDetections(frame, results);
    
    cv::namedWindow("test");
    cv::imshow("test", frame);
    cv::waitKey(0);


    return 0;
}
```
### cuda 核函数预处理与解码示例
参考手写AI的infer仓库 https://github.com/shouxieai/infer
