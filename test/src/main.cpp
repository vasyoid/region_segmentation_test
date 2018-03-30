#include "segment.h"
#include <iostream>

/**
 * Take command line arguments and segment the Object and the Background areas.
 * @param argc number of command line arguments.
 * @param argv array containing image file name, mask file name and result file name.
 * @return 0 on success, -1 otherwise.
 */
int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << "usage:\n./run image_file_name mask_file_name output_file_name" << std::endl;
        return -1;
    }
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat mask = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (image.empty() || mask.empty()) {
        std::cerr << "could not open input files" << std::endl;
        return -1;
    }
    std::vector<GraphD::node_id> nodes(static_cast<unsigned long>(image.rows * image.cols + 2));
    GraphD graph(static_cast<int>(nodes.size()), static_cast<int>(nodes.size() * 3));
    build_graph(image, mask, graph, nodes);
    graph.maxflow();
    segment(image, graph, nodes);
    //segment_new(image, graph, nodes);
    cv::imwrite(argv[3], image);
    return 0;
}