#include "AccumulatedGrad.h"
#include "TensorFactories.h"


AccumulatedGrad::AccumulatedGrad(Tensor* t, std::vector<size_t> size_of_tensor_) : tensor(t), Node(size_of_tensor_)
{
	is_leaf_node = true;
	node_type = "AccumulatedGrad";
}

std::vector<Tensor> AccumulatedGrad::backward(std::vector<Tensor>& grads)
{
	if (!(*tensor).grad) {

		(*tensor).grad = std::make_shared<Tensor>(grads[0]);
	}
	else
	{
		(*tensor).grad = std::make_shared<Tensor>(tiny_torch::add((*tensor).grad.get(), &grads[0]));
	}
	return {};

};
