#include <iostream>
#include "Tensor.h"
#include "AutoGrad.h"

int main() {
    // Define matrix dimensions
    std::vector<size_t> shape1 = { 4, 8 };
    std::vector<size_t> shape2 = { 8, 9 };
    std::vector<size_t> shape3 = { 4, 9 };

    // Initialize Tensors
    Tensor x = tiny_torch::fill(shape1, 55.0f);
    Tensor y = tiny_torch::rand(shape2);
    Tensor l = tiny_torch::rand(shape3);
    Tensor ll = tiny_torch::rand(shape3);

    // Enable gradient tracking
    x.auto_grad = true;
    y.auto_grad = true;

    // Forward Pass: n = (ll * (x @ y)) + (l + (x @ y))
    Tensor z = tiny_torch::mm(&x, &y);
    Tensor t = tiny_torch::add(&l, &z);
    Tensor k = tiny_torch::multiply(&ll, &z);
    Tensor n = tiny_torch::add(&k, &t);

    // Run Backpropagation
    tiny_torch::auto_grad::backward(n.grad_fn);

    // Access Gradients
    std::cout << "Gradient of x:\n" << x.grad_tensor()->print() << std::endl;
    std::cout << "Gradient of y:\n" << y.grad_tensor()->print() << std::endl;

    return 0;
}
