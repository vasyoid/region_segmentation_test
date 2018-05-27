#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <queue>
#include "cut.h"
#include "shift.h"
#include "grabcut.h"
#include "magic_wand.h"
#include "../edison_gpu/src/mean_shift.h"

int global_count = 1;

std::unordered_map<int, uchar> points_segmentation;

struct PointHash {
    inline std::size_t operator()(const cv::Point &p) const {
        return p.x * 10000 + p.y;
    }
};


double dist(const cv::Vec3b &vec1, const cv::Vec3b &vec2) {
    return std::sqrt(pow(vec1[0] - vec2[0], 2) + pow(vec1[1] - vec2[1], 2) + pow(vec1[2] - vec2[2], 2));
}


struct Image {
    std::unordered_map<int, cv::Point> points;
    std::unordered_map<cv::Point, int, PointHash> point_ids;
    cv::Mat image;
    cv::Mat mask;
    cv::Mat tmp;
    std::string file;
    double scale_factor = 1;

    Image() = default;

    Image(const std::string &directory, std::ifstream &input) {
        std::string mask_file;
        input >> file;
        image = cv::imread(directory + "/" + file + ".JPG", cv::IMREAD_IGNORE_ORIENTATION | cv::IMREAD_COLOR);
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
            point_ids[{x, y}] = ids[i];
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
        cv::waitKey();
    }

    void segment_next() {
        mask = cut(image, mask);
        //mask = magic_wand(image, mask);
        //mask = shift(image, mask);
    }

    static void update_points(Image img) {
        for (auto &p : img.points) {
            if (points_segmentation.count(p.first) == 0) {
                points_segmentation[p.first] = img.mask.at<uchar>(p.second * img.scale_factor);
            }
        }
    }

    void inherit_mask(Image img, cv::Mat prob) {
        mask.create(image.rows, image.cols, CV_8UC1);
        for (int i = 0; i < mask.rows; i++) {
            for (int j = 0; j < mask.cols; j++) {
                mask.at<uchar>(i, j) = 0;
            }
        }
        cv::Mat tmp;
        image.copyTo(tmp);
        for (auto &p : points) {
            if (points_segmentation.count(p.first) > 0) {
                if (points_segmentation[p.first]) {
                    if (prob.at<uchar>(p.second * scale_factor) > 2) {
                        mask.at<uchar>(p.second * scale_factor) = OBJECT_SEED;
                        cv::circle(tmp, p.second * scale_factor, 2, RED);
                    }
                } else {
                    mask.at<uchar>(p.second * scale_factor) = BACKGROUND_SEED;
                    cv::circle(tmp, p.second * scale_factor, 2, BLUE);
                }
            }
        }
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                if (prob.at<uchar>(i, j) > 150) {
                    mask.at<uchar>(i, j) = PROBABILITY_SEED;
                    cv::circle(tmp, (cv::Point){j, i}, 2, GREEN);
                }
            }
        }
        cv::imshow("inherited mask", tmp);
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
            cv::circle(((Image *)img)->tmp, cv::Point(x, y), 5, RED, cv::FILLED);
            cv::circle(((Image *)img)->mask, cv::Point(x, y), 5, 1, cv::FILLED);
            cv::imshow("create mask", ((Image *)img)->tmp);
        } else if (state == 2) {
            cv::circle(((Image *)img)->tmp, cv::Point(x, y), 5, BLUE, cv::FILLED);
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
        cv::imshow("asdasd", mask * 127);
        cv::waitKey();
    }

    void segment_first() {
        mask = grabcut(image);
    }

};

void showHist(cv::Mat hist) {
    int hist_w = 512; int hist_h = 400, histSize = 256;
    int bin_w = cvRound((double) hist_w / histSize);
    cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(0, 0, 0));
    normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    for (int i = 1; i < histSize; i++) {
        line(histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
             cv::Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
             cv::Scalar(255, 0, 0), 2, 8, 0);
    }
    cv::imshow("Histogram", histImage);
    cv::waitKey();
}

std::vector<std::vector<double>> myCalcHist(cv::Mat &img, cv::Mat &mask, bool rev = false) {
    int histSize = 256;    // bin size
    float range[] = {0, 255};
    const float *ranges[] = {range};
    cv::Mat channels[3];
    cv::split(img, channels);
    cv::Mat hists[3];
    std::vector<std::vector<double>> prob;
    prob.resize(3);
    for (int ch = 0; ch < 3; ++ch) {
        calcHist( &channels[ch], 1, 0, rev ? ~mask : mask, hists[ch], 1, &histSize, ranges);
        cv::normalize(hists[ch], hists[ch], 0, 255, cv::NORM_MINMAX, -1, cv::Mat());
        //showHist(hists[ch]);
        for (int i = 0; i < 256; ++i) {
            prob[ch].push_back(hists[ch].at<float>(i) / 255.0);
        }
    }
    return prob;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "usage:\n./multi_pic data_directory" << std::endl;
        return -1;
    }
    std::string directory = argv[1];
    std::ifstream info(directory + "/info.txt");
    std::cout << directory + "/info.txt" << std::endl;
    Image image1 = Image(directory, info);
    if (image1.empty()) {
        std::cerr << "could not open input files" << std::endl;
        return -1;
    }
    image1.scale(0.15);
    image1.segment_first();
    cv::Mat tmp(image1.image.rows, image1.image.cols, CV_8UC1, 1);
    cv::imwrite(directory + "/result/" + std::to_string(global_count++) + ".png", image1.image);
    auto prob_obj = myCalcHist(image1.image, image1.mask);
    while (info.good()) {
        Image image2(directory, info);
        if (image2.empty()) {
            std::cerr << "could not open input files" << std::endl;
            break;
        }
        image2.scale(0.15);
        for (int row = 0; row < image2.image.rows; ++row) {
            for (int col = 0; col < image2.image.cols; ++col) {
                cv::Vec3b color = image2.image.at<cv::Vec3b>(row, col);
                tmp.at<uchar>(row, col) = static_cast<uchar>(255 * prob_obj[0][color[0]] * prob_obj[1][color[1]] * prob_obj[2][color[2]]);
            }
        }
        Image::update_points(image1);
        image2.inherit_mask(image1, tmp);
        image2.segment_next();
        image2.show();
        cv::imwrite(directory + "/result/" + std::to_string(global_count++) + ".png", image2.image);
        image1 = image2;
    }
    return 0;
}