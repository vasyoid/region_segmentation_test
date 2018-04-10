#include "segment.h"
#include <iostream>

/**
 * Take command line arguments and segment the Object and the Background areas.
 * @param argc number of command line arguments.
 * @param argv array containing image file name, mask file name and result file name.
 * @return 0 on success, -1 otherwise.
 */
cv::Mat cut(cv::Mat &image, cv::Mat mask) {
    std::vector<GraphD::node_id> nodes(static_cast<unsigned long>(image.rows * image.cols + 2));
    GraphD graph(static_cast<int>(nodes.size()), static_cast<int>(nodes.size() * 3));
    build_graph(image, mask, graph, nodes);
    graph.maxflow();
    segment(image, mask, graph, nodes);
    return mask;
}