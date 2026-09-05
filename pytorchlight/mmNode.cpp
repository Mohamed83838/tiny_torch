#include "mmNode.h"
#include "TensorFactories.h"
std::vector<Tensor> mmNode::backward(std::vector<Tensor>& grads)
{
	Tensor t_x = tiny_torch::Transpose(x);
	Tensor t_y = tiny_torch::Transpose(y);
	Tensor dl_x = tiny_torch::mm(&grads[0],&t_y );
	Tensor dl_y = tiny_torch::mm(&t_x,& grads[0]);
	return { dl_x,dl_y };

}

mmNode::mmNode(Tensor* x_, Tensor* y_, std::vector<size_t> size_of_tensor_) :
	x(x_), y(y_), Node(size_of_tensor_) {
	node_type = "mmNode";
};