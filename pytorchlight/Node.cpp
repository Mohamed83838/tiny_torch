#include "Node.h"


void Node:: add_next_edge(std::shared_ptr<Edge> next_edge) {
    next_edges_.push_back(next_edge);
}

Node::Node(std::vector<size_t> size_of_tensor_)
{
	size_of_tensor = size_of_tensor_;
	sequence_nr = get_next_sequence_nr();
};
std::vector<Tensor> Node::backward( std::vector<Tensor>& grads) 
{
	return grads;
};

std::string Node::print() {
	std::string result = "Node type: " + node_type + ", Sequence number: " + std::to_string(sequence_nr);
	return result;
}