#include "CpuAllocator.h"
#include "Storage.h"
#include <iostream> 
#include <cstddef>
#include <memory> 
#include <vector> 
#include "Tensor.h"
#include <string> 
#include "TensorFactories.h"
#include "Auto_grad.h"


int main ()
{
 /*
    std::vector<size_t> shape1 = { 4, 8};
    std::vector<size_t> shape2 = { 8,9 };
    //std::vector<size_t> indices = { 2, 2 };
    //Tensor x = tiny_torch::fill(shape1,55);
    Tensor y = tiny_torch::rand(shape1);
    //Tensor z = tiny_torch::mm(x, y);
    //std::cout << x.print();
    std::cout << y.print();
    Tensor s = tiny_torch::Transpose(y);
   // std::cout << y.print();
    std::cout << s.print();*/

	std::cout << tiny_torch::auto_grad::hello();

}