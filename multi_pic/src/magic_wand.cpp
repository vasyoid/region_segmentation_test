#include "magic_wand.h"
#include <queue>
#include <iostream>
#include <opencv/cv.hpp>
#include <colors.h>
#include <stack>
#include <unordered_set>

cv::VideoWriter oVideoWriter;

typedef std::pair<int, int> MyPoint;

struct MyPointHash {
    inline std::size_t operator()(const std::pair<int,int> &v) const {
        return v.first * 10000 + v.second;
    }
};

double color_dist(cv::Vec3b &c1, cv::Vec3b &c2) {
    return (abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2])) / 3.0;
}

long long cnt = 0;

static void grow(cv::Mat &image, cv::Mat &mask, int row, int col, uchar type, double threshold, double global) {
    std::queue<MyPoint> border;
    border.push({row, col});
    while (!border.empty()) {
        int r = border.front().first;
        int c = border.front().second;
        border.pop();
        if (mask.at<uchar>(r, c) == type) {
            continue;
        }
        mask.at<uchar>(r, c) = type;
        for (int i = std::max(0, r - 1); i <= std::min(image.rows - 1, r + 1); ++i) {
            for (int j = std::max(0, c - 1); j <= std::min(image.cols - 1, c + 1); ++j) {
                if (mask.at<uchar>(i, j) != type &&
                        color_dist(image.at<cv::Vec3b>(i, j), image.at<cv::Vec3b>(r, c)) < threshold &&
                        color_dist(image.at<cv::Vec3b>(i, j), image.at<cv::Vec3b>(row, col)) < global) {
                    border.push({i, j});
                }
            }
        }
        if (type == BACKGROUND) {
            //image.at<cv::Vec3b>(r, c) = BLACK;
            if (++cnt % 100 == 1) {
                //oVideoWriter.write(image);
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

cv::Mat magic_wand(cv::Mat &image, cv::Mat &mask) {
    oVideoWriter = cv::VideoWriter("wand.avi", cv::VideoWriter::fourcc('M', 'P', '4', '2'),
                               25, cv::Size(image.cols, image.rows), true);
    cv::Mat result(mask.rows, mask.cols, CV_8UC1, OBJECT);
    for (int row = 0; row < mask.rows; ++row) {
        for (int col = 0; col < mask.cols; ++col) {
            if (mask.at<uchar>(row, col) == BACKGROUND_SEED && result.at<uchar>(row, col) != BACKGROUND) {
                grow(image, result, row, col, BACKGROUND, 4, 150);
            }
        }
    }

//    for (int row = 0; row < mask.rows; ++row) {
//        for (int col = 0; col < mask.cols; ++col) {
//            if (mask.at<uchar>(row, col) == OBJECT_SEED && result.at<uchar>(row, col) != OBJECT) {
//                grow(image, result, row, col, OBJECT, 3, 20);
//            }
//        }
//    }

    fill_holes(mask, result);

    segment(image, result);
    oVideoWriter.release();
    return result;
}