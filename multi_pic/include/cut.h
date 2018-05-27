#ifndef MULTI_PIC_CUT_H
#define MULTI_PIC_CUT_H

#include <opencv2/core/core.hpp>
#include "colors.h"
#include "mask.h"

cv::Mat cut(cv::Mat &image, cv::Mat mask);

#endif
