#include "AddNode.h"

std::vector<Tensor> AddNode::backward(const std::vector<Tensor>& grads)  
{

	Tensor dl_x = grads[0];
	Tensor dl_y = grads[0];
	return { dl_x,dl_y };
}

AddNode::AddNode() :
	Node() {
};