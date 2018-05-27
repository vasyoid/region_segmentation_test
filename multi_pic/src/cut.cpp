#include <iostream>
#include <queue>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "graph.h"
#include "cut.h"

const int SIGMA = 442;
const int DELTA = 23;

typedef Graph<double, double, double> GraphD;


/**
 * Count distance between two 3-dimensional vectors.
 * @param vec1 first vector.
 * @param vec2 second vector.
 * @return euclidean distance between points.
 */
static double dist(const cv::Vec3b &vec1, const cv::Vec3b &vec2) {
    return std::sqrt(pow(vec1[0] - vec2[0], 2) + pow(vec1[1] - vec2[1], 2) + pow(vec1[2] - vec2[2], 2));
}

/**
 * Count the cost of an edge between two pixels of intensity1 and intensity2 respectively.
 * @param intensity1 intensity of the first pixel.
 * @param intensity2 intensity of the second pixel.
 * @return the cost of the edge.
 */
static double count_cost(const cv::Vec3b &color1, const cv::Vec3b &color2) {
    double diff = (SIGMA - dist(color1, color2)) / DELTA;
    return std::exp(diff);
}

/**
 * Build a graph determined by an image.
 * The terminal nodes (SOURCE and SINK) represent the Object and the Background respectively.
 * Each of the other nodes represents a pixel.
 * There are edges between a terminal node and each non-terminal node
 * and between each pair of pixels that have a common side (imagine that pixel is a square).
 * The algorithm also recommends connecting pixels with a common vertex
 * but the bk_maxflow library does not allow nodes with more than 4 edges.
 * @param image a grayscale image determining the graph.
 * @param mask an image containing a blue area imposing hard constraints for Background pixels
 * and a red area for Object pixels.
 * @param graph Graph instance where the result will be placed.
 * @param nodes vector where the node ids of the graph will be placed.
 */
void build_graph(cv::Mat &image, cv::Mat &mask, GraphD &graph, std::vector<GraphD::node_id> &nodes) {

    // Create nodes to the graph.
    for (GraphD::node_id &x : nodes) {
        x = graph.add_node();
    }

    // Create edges between pairs of neighbour pixels.
    double max_cost = 0;
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            cv::Vec3b color1 = image.at<cv::Vec3b>(row, col);

            // Edge to the left
            if (row > 0) {
                cv::Vec3b color2 = image.at<cv::Vec3b>(row - 1, col);
                double cost = count_cost(color1, color2);
                graph.add_edge(nodes[row * image.cols + col], nodes[(row - 1) * image.cols + col], cost, cost);
                max_cost += std::abs(cost * 2);
            }

/*            if (row > 0 && col > 0) {
                cv::Vec3b color2 = image.at<cv::Vec3b>(row - 1, col - 1);
                double cost = count_cost(color1, color2);
                graph.add_edge(nodes[row * image.cols + col], nodes[(row - 1) * image.cols + col - 1], cost, cost);
                max_cost += std::abs(cost * 2);
            }

            if (row > 0 && col < image.cols - 1) {
                cv::Vec3b color2 = image.at<cv::Vec3b>(row - 1, col + 1);
                double cost = count_cost(color1, color2);
                graph.add_edge(nodes[row * image.cols + col], nodes[(row - 1) * image.cols + col + 1], cost, cost);
                max_cost += std::abs(cost * 2);
            }
*/
            // Edge up
            if (col > 0) {
                cv::Vec3b color2 = image.at<cv::Vec3b>(row, col - 1);
                double cost = count_cost(color1, color2);
                graph.add_edge(nodes[row * image.cols + col], nodes[row * image.cols + col - 1], cost, cost);
                max_cost += std::abs(cost * 2);
            }


        }
    }

    // Create edges between terminal nodes and pixel nodes.
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            int type = mask.at<uchar>(row, col);

            if (type == OBJECT_SEED) {
                // The pixel absolutely has to belong to the Object.
                graph.set_tweights(nodes[row * image.cols + col], max_cost + 1, 0);
            } else if (type == BACKGROUND_SEED) {
                // The pixel absolutely has to belong to the Background.
                graph.set_tweights(nodes[row * image.cols + col], 0, max_cost + 1);
            } else if (type == PROBABILITY_SEED) {
                // The pixel probably belongs to the Object.
                graph.set_tweights(nodes[row * image.cols + col], max_cost + 1, 0);
            }
        }
    }
}

void get_mask(cv::Mat &image, cv::Mat &old_mask, GraphD &graph, std::vector<GraphD::node_id> &nodes) {
    cv::Mat mask;
    mask.create(image.rows, image.cols, CV_8UC1);
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            if (graph.what_segment(nodes[row * image.cols + col]) == GraphD::SOURCE) {
                mask.at<uchar>(row, col) = OBJECT;
            } else {
                mask.at<uchar>(row, col) = BACKGROUND;
            }
        }
    }
    mask.copyTo(old_mask);
}

/**
 * Rebuild and image according to its graph min-cut.
 * @param image image to rebuild.
 * @param graph graph with nodes belonging to one of the terminal node's components.
 * @param nodes node ids.
 */
void segment(cv::Mat &image, cv::Mat &mask, GraphD &graph, std::vector<GraphD::node_id> &nodes) {
    get_mask(image, mask, graph, nodes);
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            if (mask.at<uchar>(row, col) == 0) {
                image.at<cv::Vec3b>(row, col) = BLACK;
            }
        }
    }
}

/**
 * Take command line arguments and segment the Object and the Background areas.
 * @param image image to segment.
 * @param mask mask with seed pixels.
 * @return segmentation mask.
 */
cv::Mat cut(cv::Mat &image, cv::Mat mask) {
    std::vector<GraphD::node_id> nodes(static_cast<unsigned long>(image.rows * image.cols + 2));
    GraphD graph(static_cast<int>(nodes.size()), static_cast<int>(nodes.size() * 3));
    build_graph(image, mask, graph, nodes);
    graph.maxflow();
    segment(image, mask, graph, nodes);
    return mask;
}