#ifndef SEGMENT_H_
#define SEGMENT_H_

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "graph.h"

const int INTENSITIES_RANGE = 256;
const int SIGMA = 442;
const int DELTA = 23;
const double LAMBDA = 0.3;

typedef Graph<double, double, double> GraphD;

void build_graph(cv::Mat &image, cv::Mat &mask, GraphD &graph, std::vector<GraphD::node_id> &nodes);
void segment(cv::Mat &image, GraphD &graph, std::vector<GraphD::node_id> &nodes);

#endif