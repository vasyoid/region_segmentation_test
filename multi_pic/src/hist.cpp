#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

void showHist(cv::MatND hist) {
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

std::vector<std::vector<double>> myCalcHist(cv::Mat &img, cv::Mat &mask) {
    int histSize = 256;    // bin size
    float range[] = {0, 255};
    const float *ranges[] = {range};
    cv::Mat channels[3];
    cv::split(img, channels);
    cv::Mat hists[3];
    long long sum = 0;
    for (int ch = 0; ch < 3; ++ch) {
        calcHist( &channels[ch], 1, 0, mask, hists[ch], 1, &histSize, ranges, true, false );
        for (int i = 0; i < 256; ++i) {
            sum += hists[ch].at<uchar>(i);
        }
    }
    std::vector<std::vector<double>> prob;
    prob.resize(3);
    for (int ch = 0; ch < 3; ++ch) {
        for (int i = 0; i < 256; ++i) {
            prob[ch].push_back(hists[ch].at<uchar>(i) / (double) sum);
        }
    }
    return prob;
}

int main(int argc, char **argv) {
    cv::Mat img = cv::imread(argv[1]);
    cv::imshow("image", img);
    cv::Mat mask(img.rows, img.cols, CV_8UC1, 1);
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = i; j < mask.cols; ++j) {
            //mask.at<cv::Vec3b>(j, i) = 0;
        }
    }
    if (img.empty()) {
        std::cout << "Could not open the image" << std::endl;
    }
    myCalcHist(img, mask);
}