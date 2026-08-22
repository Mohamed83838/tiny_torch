#include "MultNode.h"
#include "TensorFactories.h"
std::vector<Tensor> MultNode::backward( std::vector<Tensor>& grads)
{
	Tensor dl_x = tiny_torch::multiply(y, grads[0]);
	Tensor dl_y = tiny_torch::multiply(x, grads[0]);
	return { dl_x,dl_y };

}

MultNode::MultNode(Tensor& x_, Tensor& y_) :
	x(x_), y(y_),Node() {};