//
// Created by liangfuchu on 23-2-17.
//

#include "enhancement.h"

void utils::Retinex::SSR(const cv::Mat &src, cv::Mat &dst, double sigma) {
    cv::Mat L(src.size(), CV_64FC3), LLog(src.size(), CV_64FC3),
            SLog(src.size(), CV_64FC3), RLog(src.size(), CV_64FC3);
    cv::Mat src_64F;
    src.convertTo(src_64F, CV_64FC3);
    dst = cv::Mat(src.size(), CV_64FC3);
    cv::log(src_64F, SLog);
    int kSize = (int) (sigma * 3 / 2);
    kSize = kSize * 2 + 1;
    cv::GaussianBlur(src, L, cv::Size(kSize, kSize), sigma, sigma, 4);
    L.convertTo(L, CV_64FC3);
    cv::log(L, LLog);
    cv::subtract(SLog, LLog, RLog);
    cv::Mat channels[src.channels()];
    cv::split(RLog, channels);
    for (int i = 0; i < src.channels(); ++i) {
        double minVal, maxVal;
        cv::minMaxLoc(channels[i], &minVal, &maxVal);
        double scale = 255 / (maxVal - minVal);
        cv::subtract(channels[i], minVal, channels[i]);
        cv::multiply(channels[i], scale, channels[i]);
    }
    cv::merge(channels, src.channels(), dst);
    dst.convertTo(dst, CV_8UC3);
}

void utils::Retinex::MSR(const cv::Mat &src, cv::Mat &dst, std::vector<double> weights, std::vector<double> sigmas) {
    cv::Mat L(src.size(), CV_64FC3), LLog(src.size(), CV_64FC3),
            SLog(src.size(), CV_64FC3);
    cv::Mat RLog = cv::Mat::zeros(src.size(), CV_64FC3);
    cv::Mat src_64F;
    src.convertTo(src_64F, CV_64FC3);
    cv::log(src_64F, SLog);
    int n = weights.size();
    for (int i = 0; i < n; ++i) {
        cv::Mat RLogi;
        int kSize = (int) (sigmas[i] * 3 / 2);
        kSize = kSize * 2 + 1;
        cv::GaussianBlur(src, L, cv::Size(kSize, kSize), sigmas[i], sigmas[i], 4);
        L.convertTo(L, CV_64FC3);
        cv::log(L, LLog);
        cv::subtract(SLog, LLog, RLogi);
        cv::multiply(RLogi, weights[i], RLogi);
        cv::add(RLog, RLogi, RLog);
    }
    cv::Mat channels[src.channels()];
    cv::split(RLog, channels);
    for (int i = 0; i < src.channels(); ++i) {
        double minVal, maxVal;
        cv::minMaxLoc(channels[i], &minVal, &maxVal);
        double scale = 255 / (maxVal - minVal);
        cv::subtract(channels[i], minVal, channels[i]);
        cv::multiply(channels[i], scale, channels[i]);
    }
    cv::merge(channels, src.channels(), dst);
    dst.convertTo(dst, CV_8UC3);
}

void utils::gamma(const cv::Mat &src, cv::Mat &dst, double gamma) {
    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar *p = lookUpTable.ptr();
    for (int i = 0; i < 256; ++i)
        p[i] = cv::saturate_cast<uchar>(cv::pow(i / 255.0, gamma) * 255.0);
    cv::LUT(src, lookUpTable, dst);
}