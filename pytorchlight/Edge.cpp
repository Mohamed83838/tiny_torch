#include "Edge.h"


Edge::Edge(int input_nr_, std::shared_ptr<Node> next_function_)
	: input_nr(input_nr_), next_function(next_function_) {
}