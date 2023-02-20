//
// Created by liangfuchu on 23-2-16.
//
#include "match_rectification.h"

cv::Mat utils::match(const cv::Mat& image1, const cv::Mat& image2)
{
    auto surf = cv::xfeatures2d::SURF::create();
    std::vector<cv::KeyPoint> Keypoint1, Keypoint2;
    cv::Mat descriptions1, descriptions2;
    surf->detect(image1, Keypoint1);
    surf->detect(image2, Keypoint2);
    surf->compute(image1, Keypoint1, descriptions1);
    surf->compute(image2, Keypoint2, descriptions2);

    if ((descriptions1.type() != CV_32F) && (descriptions2.type() != CV_32F))
    {
        descriptions1.convertTo(descriptions1, CV_32F);
        descriptions2.convertTo(descriptions2, CV_32F);
    }

    std::vector<cv::DMatch> matches;
    cv::FlannBasedMatcher matcher;
    matcher.match(descriptions1, descriptions2, matches);
    printf("matches=%zu\n", matches.size());

    double max_dist = 0;
    double min_dist = 100;
    for (int i = 0; i < descriptions1.rows; ++i)
    {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptions1.rows; i++)
    {
        if (matches[i].distance < 0.5 * max_dist) good_matches.push_back(matches[i]);
    }

    std::vector<cv::Point2f> srcPoint(good_matches.size()), dstPoint(good_matches.size());
    for (int i = 0; i < good_matches.size(); ++i)
    {
        srcPoint[i] = Keypoint1[good_matches[i].queryIdx].pt;
        dstPoint[i] = Keypoint2[good_matches[i].trainIdx].pt;
    }
    cv::Mat H = cv::findHomography(srcPoint, dstPoint);
    return H;
}


cv::Mat utils::rectification(const cv::Mat& image, cv::Mat H, cv::Size Size)
{
    cv::Mat warp_image;
    cv::warpPerspective(image, warp_image, H, Size);
    return warp_image;
}