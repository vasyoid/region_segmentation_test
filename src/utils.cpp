#include <iostream>
#include "segment.h"

/**
 * Calculate probabilities of intensities (0 to 255) based on pixels covered by a mask.
 * The probability of an intensity is proportional to the number of pixels of this intensity.
 * @param image pixels source.
 * @param mask mask defining which pixels
 * @return vector of size 256 containing probabilities.
 */
static std::vector<double> calc_probabilities(cv::Mat &image, cv::Mat &mask) {

    // Set parameters to calculate a histogram of the given area.
    int hist_size = INTENSITIES_RANGE;
    float range[] = {0, INTENSITIES_RANGE};
    const float* hist_range = {range};

    // Calculate histogram and scale the values to be in range [0, 255].
    cv::MatND hist;
    cv::calcHist(&image, 1, nullptr, mask, hist, 1, &hist_size, &hist_range, true, false);
    cv::normalize(hist, hist, 0, INTENSITIES_RANGE, cv::NORM_MINMAX, -1);

    // Count the total number of pixels in the area.
    double total = 0;
    for (int i = 0; i < hist_size; ++i) {
        total += hist.at<float>(i);
    }

    // Count the probability for each range
    // as a ratio of an intensity inclusion to the total number of pixels.
    std::vector<double> result;
    for (int i = 0; i < hist_size; ++i) {
        result.push_back(hist.at<float>(i) / total);
    }
    return result;
}

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
    double diff = SIGMA - std::abs(dist(color1, color2) / 20);
    return std::exp(std::abs(diff)) * (diff < 0 ? -1 : 1);
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

    // Calculate probabilities of intensities for the Object and Background cases.
    std::vector<cv::Mat> mask_layers;
    cv::split(mask, mask_layers);
    cv::bitwise_not(mask_layers[0], mask_layers[0]);
    cv::bitwise_not(mask_layers[2], mask_layers[2]);
    //std::vector<double> prob_obj = calc_probabilities(image, mask_layers[0]);
    //std::vector<double> prob_bkg = calc_probabilities(image, mask_layers[2]);

    // Create edges between terminal nodes and pixel nodes.
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            cv::Vec3b color = mask.at<cv::Vec3b>(row, col);

            if (compareColors(color, {0, 0, 255})) {
                // The pixel absolutely has to belong to the Object.
                graph.set_tweights(nodes[row * image.cols + col], max_cost + 1, 0);
                //std:: cout << row * image.cols + col << " -- S: " << max_cost + 1 << std::endl;
            } else if (compareColors(color, {255, 0, 0})) {
                // The pixel absolutely has to belong to the Background.
                graph.set_tweights(nodes[row * image.cols + col], 0, max_cost + 1);
                //std:: cout << row * image.cols + col << " -- T: " << max_cost + 1 << std::endl;
            }
                /*
            else {
                // There are no hard constraints for the pixel.
                double cost_obj = LAMBDA * -std::log(prob_obj[image.at<uchar>(row, col)]);
                double cost_bkg = LAMBDA * -std::log(prob_bkg[image.at<uchar>(row, col)]);
                graph.set_tweights(nodes[row * image.cols + col], cost_bkg, cost_obj);
            }
                 */
        }
    }
}

/**
 * Rebuild and image according to its graph min-cut.
 * @param image image to rebuild.
 * @param graph graph with nodes belonging to one of the terminal node's components.
 * @param nodes node ids.
 */
void segment(cv::Mat &image, GraphD &graph, std::vector<GraphD::node_id> &nodes) {
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            if (graph.what_segment(nodes[row * image.cols + col]) == GraphD::SOURCE) {
                image.at<uchar>(row, col) = 255;
            } else {
                image.at<uchar>(row, col) = 0;
            }
        }
    }
}
