#include "Node.h"


void Node:: add_next_edge(std::shared_ptr<Edge> next_edge) {
    next_edges_.push_back(next_edge);
}

Node::Node()
{
	sequence_nr = get_next_sequence_nr();
};
std::vector<Tensor> Node::backward(const std::vector<Tensor>& grads) 
{
	return grads;
};
