#pragma once
#include "Node.h"


class AccumulatedGrad : public Node {
public:
    Tensor& tensor; 

    AccumulatedGrad(Tensor& t);

    void backward( std::vector<Tensor>& grads) ;
};