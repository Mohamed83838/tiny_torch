#pragma once
#include "Node.h"


class AccumulatedGrad : public Node {
public:
    Tensor* tensor; 

    AccumulatedGrad(Tensor* t, std::vector<size_t> size_of_tensor_);

    std::vector<Tensor> backward( std::vector<Tensor>& grads) override;
};