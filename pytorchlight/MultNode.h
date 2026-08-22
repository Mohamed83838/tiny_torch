#pragma once
#include "Node.h"

class MultNode:public Node 
{
public:
	 Tensor& x;
	 Tensor& y;
	MultNode( Tensor& x,  Tensor& y);
	std::vector<Tensor> backward( std::vector<Tensor>& grads) ;
};