#pragma once
#include "Node.h"

class MultNode:public Node 
{
public:
	 Tensor* x;
	 Tensor* y;
	 std::vector<Tensor> backward( std::vector<Tensor>& grads) override ;
	MultNode( Tensor* x,  Tensor* y, std::vector<size_t> size_of_tensor_);
	
};