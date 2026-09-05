#pragma once
#include "Node.h"


class AddNode :public Node {
public:
	std::vector<Tensor> backward( std::vector<Tensor>& grads) override ;
	AddNode(std::vector<size_t> size_of_tensor_);
	~AddNode() = default;
};

