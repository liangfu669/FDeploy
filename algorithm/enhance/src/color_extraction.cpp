//
// Created by liangfuchu on 23-2-16.
//
#include "color_extraction.h"




cv::Mat utils::HsvExtraction::extraction_color(cv::Mat image, const std::string &color) const
{
    cv::Mat image_hsv;
    cv::cvtColor(image, image_hsv, cv::COLOR_BGR2HSV);
    if (color == "black")
        cv::inRange(image_hsv, black.first, black.second, image);
    else if (color == "grey")
        cv::inRange(image_hsv, grey.first, green.second, image);
    else if (color == "green")
        cv::inRange(image_hsv, green.first, green.second, image);
    else if (color == "white")
        cv::inRange(image_hsv, white.first, white.second, image);
    else if (color == "orange")
        cv::inRange(image_hsv, orange.first, orange.second, image);
    else if (color == "yellow")
        cv::inRange(image_hsv, yellow.first, yellow.second, image);
    else if (color == "cyan")
        cv::inRange(image_hsv, cyan.first, cyan.second, image);
    else if (color == "blue")
        cv::inRange(image_hsv, blue.first, blue.second, image);
    else if (color == "purple")
        cv::inRange(image_hsv, purple.first, purple.second, image);
    else if (color == "red")
    {
        cv::Mat mask1, mask2;
        cv::inRange(image_hsv, red1.first, red1.second, mask1);
        cv::inRange(image_hsv, red2.first, red2.second, mask2);
        cv::bitwise_or(mask1, mask2, image);
    }
    return image;
}


