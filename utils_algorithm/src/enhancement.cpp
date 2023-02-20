//
// Created by liangfuchu on 23-2-17.
//

#include "enhancement.h"


cv::Mat utils::Retinex::SSR(cv::Mat src, double sigma)
{
    cv::Mat src_log, gauss, gauss_log;
    Eigen::MatrixXf s;
    cv::cv2eigen(src, s);
    cv::log(src, src_log);
    return src;
}