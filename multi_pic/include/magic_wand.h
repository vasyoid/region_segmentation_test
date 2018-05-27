#ifndef MAGIC_WAND_H_
#define MAGIC_WAND_H_

#include <opencv2/core/core.hpp>
#include "mask.h"

cv::Mat magic_wand(cv::Mat &image, cv::Mat &mask);

#endif