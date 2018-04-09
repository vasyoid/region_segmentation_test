#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "grabcut.h"
#include "cut.h"
#include <fstream>
#include <iostream>
#include <unordered_map>

struct Image {
    std::unordered_map<int, cv::Point> points;
    cv::Mat image;
    cv::Mat mask;
    std::string file;
    double scale_factor = 1;

    Image(const std::string &directory, std::ifstream &input) {
        input >> file;
        image = cv::imread(directory + "/" + file + ".JPG", cv::IMREAD_COLOR);
        std::cout << directory + "/" + file + ".JPG" << std::endl;
        int point_n;
        input >> point_n;
        std::vector<int> ids;
        for (int i = 0; i < point_n; ++i) {
            int id;
            input >> id;
            ids.push_back(id);
        }
        for (int i = 0; i < point_n; ++i) {
            int x, y;
            input >> x >> y;
            points[ids[i]] = {x, y};
        }
    }

    void scale(double factor) {
        scale_factor *= factor;
        cv::resize(image, image, cv::Size((int)(image.cols * factor),
                                          (int)(image.rows * factor)), cv::INTER_CUBIC);
    }

    bool empty() {
        return image.empty();
    }

    void show() {
        cv::imshow("", image);
        cv::waitKey();
    }

    void show_mask() {
        cv::Mat tmp = image;
        for (int row = 0; row < tmp.rows; ++row) {
            for (int col = 0; col < tmp.rows; ++col) {
                if (mask.at<uchar>(col, row) == 0) {
                    tmp.at<cv::Vec3b>(col, row) = {0, 0, 0};
                }
            }
        }
        cv::imshow("", tmp);
        cv::waitKey();
    }

    void segment_first() {
        mask = segmentate(image);
    }

    void segment_next() {
        mask = cut(image, mask);
    }

    void inherit_mask(Image img) {
        mask.create(image.rows, image.cols, CV_8UC1);
        for (auto &p : img.points) {
            if (points.count(p.first) > 0) {
                if (img.mask.at<uchar>(p.second * scale_factor)) {
                    mask.at<uchar>(points[p.first] * scale_factor) = 1;
                } else {
                    mask.at<uchar>(points[p.first] * scale_factor) = 2;
                }
            }
        }
    }

    void save(std::string directory) {
        cv::imwrite(directory + "/" + file + "-2.JPG", image);
    }

};

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "usage:\n./multi_pic data_directory" << std::endl;
        return -1;
    }
    std::string directory = argv[1];
    std::ifstream info(directory + "/info.txt");
    std::cout << directory + "/info.txt" << std::endl;
    Image image1(directory, info);
    image1.scale(0.2);
    if (image1.empty()) {
        std::cerr << "could not open input files" << std::endl;
        return -1;
    }
    image1.segment_first();
    while (!info.eof()) {
        Image image2(directory, info);
        if (image2.empty()) {
            std::cerr << "could not open input files" << std::endl;
            continue;
        }
        image2.scale(0.2);
        image2.inherit_mask(image1);
        image2.segment_next();
        image2.show();
        std::swap(image1, image2);
    }
    return 0;
}