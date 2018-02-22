#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main() {
    cv::Mat image = cv::imread("../res/img.png", cv::IMREAD_COLOR);
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE);
    for (auto x = image.begin<cv::Vec3b>(); x != image.end<cv::Vec3b>(); ++x) {
        (*x)[0] = static_cast<uchar>(((*x)[0] + (*x)[1] + (*x)[2]) / 3);
        (*x)[2] = (*x)[1] = (*x)[0];
    }
    imshow("Display window", image);
    cv::waitKey(0);
    cv::imwrite("../res/out.png", image);
    return 0;
}