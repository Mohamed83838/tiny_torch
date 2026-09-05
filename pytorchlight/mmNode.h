#pragma once
#pragma once
#include "Node.h"

class mmNode :public Node
{
public:
	Tensor* x;
	Tensor* y;
	mmNode(Tensor* x, Tensor* y, std::vector<size_t> size_of_tensor_);
	std::vector<Tensor> backward(std::vector<Tensor>& grads) override;
};