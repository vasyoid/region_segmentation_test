#include <opencv2/imgcodecs.hpp>
#include <opencv/cv.hpp>
#include "magic_wand.h"

int main(int argc, char **argv) {
    cv::Mat image = cv::imread(argv[1]);
    cv::resize(image, image, {}, 0.22, 0.22);
    cv::Mat mask(image.rows, image.cols, CV_8UC1, EMPTY_SEED);
    mask.at<uchar>(10, 100) = BACKGROUND_SEED;
    mask = magic_wand(image, mask);
    cv::imshow("window", image);
    cv::waitKey();
}