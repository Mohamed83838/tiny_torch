#pragma once
#include "Node.h"
class AddNode :public Node {
public:
	std::vector<Tensor> backward(const std::vector<Tensor>& grads) override ;
	AddNode();
	~AddNode() = default;
};