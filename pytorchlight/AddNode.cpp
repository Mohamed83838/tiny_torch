#include "AddNode.h"

std::vector<Tensor> AddNode::backward( std::vector<Tensor>& grads)  
{

	Tensor dl_x = grads[0];
	Tensor dl_y = grads[0];
	return { dl_x,dl_y };
}

AddNode::AddNode(std::vector<size_t> size_of_tensor_) :
	Node(size_of_tensor_) {
	node_type = "AddNode";
};