//
// Created by liangfuchu on 23-2-17.
//

#include "enhancement.h"

void utils::Retinex::SSR(cv::Mat src, cv::Mat &dst, double sigma) {
    cv::Mat L(src.size(), CV_32FC3), LLog(src.size(), CV_32FC3),
            SLog(src.size(), CV_32FC3), RLog(src.size(), CV_32FC3);
    dst = cv::Mat(src.size(), CV_32FC3);
    // 求Log[S(x,y)]
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            for (int k = 0; k < 3; ++k) {
                float value = src.at<cv::Vec3b>(i, j)[k];
                if (value <= 0.01) value = 0.01;
                SLog.at<cv::Vec3f>(i, j)[k] = log10(value);
            }
        }
    }
    // 求L(x,y)
    int kSize = (int) (sigma * 3 / 2);
    kSize = kSize * 2 + 1;
    cv::GaussianBlur(src, L, cv::Size(kSize, kSize), sigma, sigma, 4);
    // 求Log[L(x,y)]
    for (int i = 0; i < L.rows; ++i) {
        for (int j = 0; j < L.cols; ++j) {
            for (int k = 0; k < 3; ++k) {
                float value = L.at<cv::Vec3b>(i, j)[k];
                if (value <= 0.01) value = 0.01;
                LLog.at<cv::Vec3f>(i, j)[k] = log10(value);
            }
        }
    }
    // 求Log[R(x,y)]
    float vMax[3] = {0, 0, 0};
    float vMin[3] = {0, 0, 0};
    for (int i = 0; i < L.rows; ++i) {
        for (int j = 0; j < L.cols; ++j) {
            for (int k = 0; k < 3; ++k) {
                auto value = SLog.at<cv::Vec3f>(i, j)[k] - LLog.at<cv::Vec3f>(i, j)[k];
                RLog.at<cv::Vec3f>(i, j)[k] = value;
                if (value > vMax[k]) vMax[k] = value;
                else if (value < vMin[k]) vMin[k] = value;
            }
        }
    }
    // 求R(x,y)
    for (int i = 0; i < RLog.rows; ++i) {
        for (int j = 0; j < RLog.cols; ++j) {
            for (int k = 0; k < 3; ++k) {
                float value = RLog.at<cv::Vec3f>(i, j)[k];
                dst.at<cv::Vec3f>(i, j)[k] = cv::saturate_cast<float>((value - vMin[k]) * 255 / (vMax[k] - vMin[k]));
            }
        }
    }
    dst.convertTo(dst, CV_8UC3);
}