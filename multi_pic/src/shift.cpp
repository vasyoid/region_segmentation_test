#include "../edison_gpu/src/mean_shift.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv/cv.hpp>
#include <queue>
#include <mask.h>
#include <unordered_set>
#include "../edison_gpu/thirdparty/cimg/src/images.h"
#include "colors.h"

typedef std::pair<int, int> MyPoint;

struct MyPointHash {
    inline std::size_t operator()(const std::pair<int,int> &v) const {
        return v.first * 10000 + v.second;
    }
};

static void grow(cv::Mat &mask, cv::Mat &result, int row, int col, uchar type) {
    std::queue<MyPoint> border;
    border.push({row, col});
    while (!border.empty()) {
        int r = border.front().first;
        int c = border.front().second;
        border.pop();
        if (result.at<uchar>(r, c) == type) {
            continue;
        }
        result.at<uchar>(r, c) = type;
        for (int i = std::max(0, r - 1); i <= std::min(result.rows - 1, r + 1); ++i) {
            for (int j = std::max(0, c - 1); j <= std::min(result.cols - 1, c + 1); ++j) {
                if (result.at<uchar>(i, j) != type && mask.at<uchar>(i, j) != BORDER) {
                    border.push({i, j});
                } else {
                    result.at<uchar>(i, j) = type;
                }
            }
        }

    }
}

static void get_area(std::unordered_set<MyPoint, MyPointHash> &area, cv::Mat &mask, int row, int col) {
    std::queue<MyPoint> border;
    border.push({row, col});
    while (!border.empty()) {
        int r = border.front().first;
        int c = border.front().second;
        border.pop();
        if (area.count({r, c}) > 0) {
            continue;
        }
        area.insert({r, c});
        for (int i = std::max(0, r - 1); i <= std::min(mask.rows - 1, r + 1); ++i) {
            for (int j = std::max(0, c - 1); j <= std::min(mask.cols - 1, c + 1); ++j) {
                if (mask.at<uchar>(i, j) == OBJECT) {
                    border.push({i, j});
                }
            }
        }
    }
}

static void segment(cv::Mat &image, cv::Mat &mask) {
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            if (mask.at<uchar>(row, col) == BACKGROUND) {
                image.at<cv::Vec3b>(row, col) = BLACK;
            }
        }
    }
}

static void fill_holes(cv::Mat &mask, cv::Mat &result) {
    std::unordered_set<MyPoint, MyPointHash> object;
    for (int row = 0; row < result.rows; ++row) {
        for (int col = 0; col < result.cols; ++col) {
            if (mask.at<uchar>(row, col) == OBJECT_SEED && !object.count({row, col})) {
                get_area(object, result, row, col);
            }
        }
    }
    for (int row = 0; row < result.rows; ++row) {
        for (int col = 0; col < result.cols; ++col) {
            if (!object.count({row, col})) {
                result.at<uchar>(row, col) = BACKGROUND;
            }
        }
    }
}

cv::Mat shift(cv::Mat &image, cv::Mat &mask) {
    SegmentedRegions regions = meanShiftSegmentation(image.data, image.cols, image.rows, image.channels(),
                                                     8, 5, 200, MULTITHREADED_SPEEDUP, true);
    cv::Mat tmp(image.rows, image.cols, CV_8UC1, BACKGROUND);
    for (size_t i = 0; i < regions.getNumRegions(); ++i) {
        std::vector<PixelPosition> border = regions.getRegionBorder(i);
        for (size_t j = 0; j < border.size(); ++j) {
            tmp.at<uchar>(border[j].row, border[j].column) = BORDER;
        }
    }
    cv::imshow("borders", tmp);
    cv::waitKey();
    cv::Mat result(mask.rows, mask.cols, CV_8UC1, OBJECT);
    for (int row = 0; row < mask.rows; ++row) {
        for (int col = 0; col < mask.cols; ++col) {
            if (mask.at<uchar>(row, col) == BACKGROUND_SEED && result.at<uchar>(row, col) != BACKGROUND) {
                grow(tmp, result, row, col, BACKGROUND);
            }
        }
    }
    fill_holes(mask, result);
    segment(image, result);
    cv::imshow("segmentation", result * 127);
    return result;
}
