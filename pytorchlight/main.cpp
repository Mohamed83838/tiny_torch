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

int main() {
    try {
        // Fix: shape1's columns (8) must equal shape2's rows (8)
        std::vector<size_t> shape1 = { 4, 8 };
        std::vector<size_t> shape2 = { 8, 9 };
        std::vector<size_t> shape3 = { 4, 9 };

        Tensor x = tiny_torch::fill(shape1, 55);
        Tensor y = tiny_torch::rand(shape2);
        Tensor l = tiny_torch::rand(shape3);
        Tensor ll = tiny_torch::rand(shape3);
        x.auto_grad = true;
        y.auto_grad = true;
        l.auto_grad = true;
        ll.auto_grad = true;

        // Matrix multiplication (4x8) * (8x9) = (4x9)
        Tensor z = tiny_torch::mm(&x, &y);
		Tensor t = tiny_torch::add(&l,&z);
        Tensor k = tiny_torch::multiply(&ll, &z);
        Tensor n = tiny_torch::add(&k, &t);

		std::cout << x.print() << std::endl;
        std::cout << y.print() << std::endl;
        std::cout << l.print() << std::endl;
        std::cout << ll.print() << std::endl;
        std::cout << z.print() << std::endl;
        std::cout << t.print() << std::endl;
		std::cout << k.print() << std::endl;
        std::cout << n.print() << std::endl;
        
tiny_torch::auto_grad::backward(n.grad_fn);
std::cout << "Gradient of x:\n" << x.grad_tensor()->print() << std::endl;
std::cout << "Gradient of y:\n" << y.grad_tensor()->print() << std::endl;

       
    }
    
    catch (const std::exception& e) {
        // Catches any dimension mismatch exceptions instead of calling abort()
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;

}