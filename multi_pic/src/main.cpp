#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "grabcut.h"
#include "cut.h"
#include <fstream>
#include <iostream>
#include <unordered_map>

int global_count = 1;

std::unordered_map<int, uchar> points_segmentation;

struct Image {
    std::unordered_map<int, cv::Point> points;
    cv::Mat image;
    cv::Mat mask;
    cv::Mat tmp;
    std::string file;
    double scale_factor = 1;

    Image(const std::string &directory, std::ifstream &input) {
        std::string mask_file;
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
        if (!mask.empty()) {
            cv::resize(mask, mask, cv::Size(image.cols, image.rows), cv::INTER_CUBIC);
        }
    }

    bool empty() {
        return image.empty();
    }

    void show() {
        cv::imshow("result", image);
        cv::imwrite(std::to_string(global_count++) + ".png", image);
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
        cv::Mat tmp;
        image.copyTo(tmp);
        for (auto &p : img.points) {
            if (points_segmentation.count(p.first) == 0) {
                points_segmentation[p.first] = img.mask.at<uchar>(p.second * scale_factor);
            }
        }
        for (auto &p : points_segmentation) {
            if (points.count(p.first) > 0) {
                if (p.second) {
                    mask.at<uchar>(points[p.first] * scale_factor) = 1;
                    cv::circle(mask, points[p.first] * scale_factor, 5, 1, cv::FILLED);
                    cv::circle(tmp, points[p.first] * scale_factor, 3, cv::Scalar(0, 0, 255));
                } else {
                    cv::circle(mask, points[p.first] * scale_factor, 5, 2, cv::FILLED);
                    cv::circle(tmp, points[p.first] * scale_factor, 3, cv::Scalar(255, 0, 0));
                }
            }
        }
        cv::imshow("inherited mask", tmp);
        cv::imwrite(std::to_string(global_count++) + ".png", tmp);
        cv::waitKey();
    }

    static void mouseClick(int event, int x, int y, int flags, void* img) {
        static int state = 0;
        switch (event) {
            case cv::EVENT_LBUTTONDOWN:
                state = 1;
                break;
            case cv::EVENT_RBUTTONDOWN:
                state = 2;
                break;
            case cv::EVENT_LBUTTONUP:
            case cv::EVENT_RBUTTONUP:
                state = 0;
                break;
            case cv::EVENT_MOUSEMOVE:
                break;
            default:
                return;
        }
        if (state == 1) {
            cv::circle(((Image *)img)->tmp, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), cv::FILLED);
            cv::circle(((Image *)img)->mask, cv::Point(x, y), 5, 1, cv::FILLED);
            cv::imshow("create mask", ((Image *)img)->tmp);
        } else if (state == 2) {
            cv::circle(((Image *)img)->tmp, cv::Point(x, y), 5, cv::Scalar(255, 0, 0), cv::FILLED);
            cv::circle(((Image *)img)->mask, cv::Point(x, y), 5, 2, cv::FILLED);
            cv::imshow("create mask", ((Image *)img)->tmp);
        }
    }

    void create_mask() {
        image.copyTo(tmp);
        mask.create(image.rows, image.cols, CV_8UC1);
        cv::imshow("create mask", image);
        cv::setMouseCallback("create mask", mouseClick, this);
        cv::waitKey();
        cv::destroyWindow("create mask");
    }

    void make_mask_smooth() {
        const int area = 5;
        mask.copyTo(tmp);
        for (int row = 0; row < mask.rows; ++row) {
            for (int col = 0; col < mask.cols; ++col) {
                int alike_points = 0;
                for (int i = std::max(row - area, 0); i < std::min(row + area + 1, mask.rows); ++i) {
                    for (int j = std::max(col - area, 0); j < std::min(col + area + 1, mask.cols); ++j) {
                        alike_points += (tmp.at<uchar>(i, j) == mask.at<uchar>(row, col));
                    }
                }
                if (alike_points < area * area) {
                    mask.at<uchar>(row, col) = (uchar)(1 - mask.at<uchar>(row, col));
                    image.at<cv::Vec3b>(row, col) = mask.at<uchar>(row, col) ? cv::Vec3b(255, 255, 255) : cv::Vec3b(0, 0, 0);
                }
            }
        }
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
    image1.create_mask();
    image1.segment_next();
    image1.show();
    while (  info.good()) {
        Image image2(directory, info);
        if (image2.empty()) {
            std::cerr << "could not open input files" << std::endl;
            break;
        }
        image2.scale(0.2);
        image2.inherit_mask(image1);
        image2.segment_next();
        //image2.make_mask_smooth();
        image2.show();
        std::swap(image1, image2);
    }
    return 0;
}