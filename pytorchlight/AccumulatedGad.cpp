#include "AccumulatedGrad.h"
#include "TensorFactories.h"


AccumulatedGrad::AccumulatedGrad(Tensor& t) : tensor(t),Node() {}

void AccumulatedGrad::backward(std::vector<Tensor>& grads) 
{
	*tensor.grad = tiny_torch::add(*tensor.grad, grads[0]);


};
