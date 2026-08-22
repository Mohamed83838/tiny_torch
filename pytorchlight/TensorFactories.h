#pragma once
#include "Tensor.h"
namespace tiny_torch 
{
    Tensor zeros(const std::vector<std::size_t>& sizes);
    Tensor ones(const std::vector<std::size_t>& sizes);
    Tensor fill(const std::vector<std::size_t>& sizes,double fillvalue);
    Tensor rand(const std::vector<std::size_t>& sizes);
    Tensor randn(const std::vector<std::size_t>& sizes);
    Tensor add(Tensor& x , Tensor& y);
    Tensor mm( Tensor& x, Tensor& y);
    Tensor Transpose(Tensor& x);

    //Tensor fromarray(std::vector<double_t> input);

    Tensor multiply(Tensor& x, Tensor& y);
}