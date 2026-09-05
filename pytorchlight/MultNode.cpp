#include "MultNode.h"
#include "TensorFactories.h"
#include <iostream>
std::vector<Tensor> MultNode::backward( std::vector<Tensor>& grads)
{
	
	
	Tensor dl_x = tiny_torch::multiply(y, &grads[0]);
	Tensor dl_y = tiny_torch::multiply(x, &grads[0]);
	return { dl_x,dl_y };

}

MultNode::MultNode(Tensor* x_, Tensor* y_, std::vector<size_t> size_of_tensor_) :
	x(x_), y(y_), Node(size_of_tensor_) {
	node_type = "MultNode";
};