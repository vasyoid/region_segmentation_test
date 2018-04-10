#include <iostream>
#include "segment.h"

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
 * Check if two colors are similar.
 * @param color1 first color.
 * @param color2 second color.
 * @return true if the distance between the colors is small, false otherwise.
 */
static bool compareColors(const cv::Vec3b &color1, const cv::Vec3b &color2) {
    return dist(color1, color2) < 50;
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

            if (type == 1) {
                // The pixel absolutely has to belong to the Object.
                graph.set_tweights(nodes[row * image.cols + col], max_cost + 1, 0);
            } else if (type == 2) {
                // The pixel absolutely has to belong to the Background.
                graph.set_tweights(nodes[row * image.cols + col], 0, max_cost + 1);
            }
        }
    }
}

void go(cv::Mat &mask, int fst_row, int fst_col, int color) {
    std::vector<std::pair<int, int>> stack;
    stack.emplace_back(std::make_pair(fst_row, fst_col));
    while (!stack.empty()) {
        auto cur = stack.back();
        stack.pop_back();
        int row = cur.first;
        int col = cur.second;
        if (row < 0 || col < 0 || row >= mask.rows || col >= mask.cols) {
            continue;
        }
        auto current = mask.at<uchar>(row, col);
        if (current < 2 || current != color) {
            continue;
        }
        mask.at<uchar>(row, col) -= 2;
        for (int i = row - 1; i < row + 2; ++i) {
            for (int j = col - 1; j < col + 2; ++j) {
                stack.emplace_back(std::make_pair(i, j));
            }
        }
    }
}

void get_mask(cv::Mat &image, cv::Mat &old_mask, GraphD &graph, std::vector<GraphD::node_id> &nodes) {
    cv::Mat mask;
    mask.create(image.rows, image.cols, CV_8UC1);
    int obj_row = 0, obj_col = 0, bgr_row = 1, bgr_col = 1;
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            if (graph.what_segment(nodes[row * image.cols + col]) == GraphD::SOURCE) {
                mask.at<uchar>(row, col) = 1;
            } else {
                mask.at<uchar>(row, col) = 0;
            }
            if (old_mask.at<uchar>(row, col) == 1) {
                obj_row = row;
                obj_col = col;
            } else if (old_mask.at<uchar>(row, col) == 2) {
                bgr_row = row;
                bgr_col = col;
            }
        }
    }
//    go(mask, bgr_row, bgr_col, 2);
//    go(mask, obj_row, obj_col, 3);
//    for (int row = 0; row < image.rows; ++row) {
//        for (int col = 0; col < image.cols; ++col) {
//            if (mask.at<uchar>(row, col) > 1) {
//                mask.at<uchar>(row, col) = 3 - mask.at<uchar>(row, col);
//                uchar res = mask.at<uchar>(row, col);
//                assert(res == 0 || res == 1);
//            }
//        }
//    }
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
                image.at<cv::Vec3b>(row, col) = {0, 0, 0};
            }
        }
    }
}
