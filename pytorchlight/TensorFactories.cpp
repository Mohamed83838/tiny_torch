#include "TensorFactories.h"
#include "TensorImpl.h"
#include <random>
#include "Storage.h"
#include <numeric>
#include <algorithm>
#include <memory>
#include "Tensor.h"
#include "AddNode.h"
#include "MultNode.h"
#include "mmNode.h"
#include <iostream>

namespace tiny_torch {

    //test namespace

    void test_namespace() {
        std::cout << "Namespace tiny_torch is working!" << std::endl;
	}

    // Passed by const reference to avoid copying
    static std::size_t calculate_numel(const std::vector<std::size_t>& sizes) {
        std::size_t dim = sizes.size();
        std::size_t res = 1;
        for (size_t i = 0; i < dim; i++) {
            res *= sizes[i];
        }
        return res;
    }

    Tensor zeros(const std::vector<std::size_t>& sizes) {
        std::size_t numel = calculate_numel(sizes);
        std::shared_ptr<Storage> storage = std::make_shared<Storage>(numel);
        if (storage->data()) {
            double* data_ptr = static_cast<double*>(storage->data());
            std::fill_n(data_ptr, numel, 0.0);
        }
        std::shared_ptr<TensorImpl> impl = std::make_shared<TensorImpl>(std::move(storage), sizes);
        return Tensor(std::move(impl));
    }

    Tensor ones(const std::vector<std::size_t>& sizes) {
        std::size_t numel = calculate_numel(sizes);
        std::shared_ptr<Storage> storage = std::make_shared<Storage>(numel);
        if (storage->data()) {
            // FIXED: Added static_cast
            double* data_ptr = static_cast<double*>(storage->data());
            std::fill_n(data_ptr, numel, 1.0);
        }
        std::shared_ptr<TensorImpl> impl = std::make_shared<TensorImpl>(std::move(storage), sizes);
        return Tensor(std::move(impl));
    }

    Tensor fill(const std::vector<std::size_t>& sizes, double value) {
        std::size_t numel = calculate_numel(sizes);
        std::shared_ptr<Storage> storage = std::make_shared<Storage>(numel);
        if (storage->data()) {
            // FIXED: Added static_cast
            double* data_ptr = static_cast<double*>(storage->data());
            std::fill_n(data_ptr, numel, value);
        }
        std::shared_ptr<TensorImpl> impl = std::make_shared<TensorImpl>(std::move(storage), sizes);
        return Tensor(std::move(impl));
    }

    Tensor rand(const std::vector<std::size_t>& sizes) {
        std::size_t numel = calculate_numel(sizes);
        std::shared_ptr<Storage> storage = std::make_shared<Storage>(numel);

        if (storage->data()) {
            double* data_ptr = static_cast<double*>(storage->data());

            // Static engine ensures we don't reseed and get the same numbers every call
            static std::random_device rd;
            static std::mt19937_64 gen(rd());
            std::uniform_real_distribution<double> dist(0.0, 1.0);

            // std::generate_n runs an optimized loop calling the lambda for each element
            std::generate_n(data_ptr, numel, [&]() { return dist(gen); });
        }

        std::shared_ptr<TensorImpl> impl = std::make_shared<TensorImpl>(std::move(storage), sizes);
        return Tensor(std::move(impl));
    }

    Tensor randn(const std::vector<std::size_t>& sizes) {
        std::size_t numel = calculate_numel(sizes);
        auto storage = std::make_shared<Storage>(numel);

        if (storage->data()) {
            double* data_ptr = static_cast<double*>(storage->data());

            static std::random_device rd;
            static std::mt19937_64 gen(rd());
            std::normal_distribution<double> dist(0.0, 1.0);

            std::generate_n(data_ptr, numel, [&]() { return dist(gen); });
        }

        std::shared_ptr<TensorImpl> impl = std::make_shared<TensorImpl>(std::move(storage), sizes);
        return Tensor(std::move(impl));
    }
	// Addition of two tensors with the same size
    Tensor add(Tensor* x, Tensor* y)

    {

        //get sizes
        const std::vector<std::size_t>& sizes_x = x->sizes();
        const std::vector<std::size_t>& sizes_y = y->sizes();
        if (sizes_x != sizes_y)
        {

            throw std::runtime_error("matrices must  have the same size for addition");
        }
        //create the output tensor
        Tensor res = zeros(sizes_x);
        //get the data and convert it:
        double* data_x = x->data_ptr();
        double* data_y = y->data_ptr();
        double* data_res = res.data_ptr();
        //initiate number of element and number of dimention
        size_t numel = x->numel();
        size_t numd = x->sizes().size();
        //initiate the indices :
        std::vector<size_t> indices(numd, 0);
        //the actual loop 
        for (size_t i = 0; i < numel; i++)
        {
            size_t pos_x = x->get_flatten_indice(indices);
            size_t pos_y = y->get_flatten_indice(indices);
            size_t pos_out = res.get_flatten_indice(indices);

            data_res[pos_out] = data_x[pos_x] + data_y[pos_y];

            for (std::size_t j = numd; j > 0; j--)
            {
                indices[j - 1]++;
                if (indices[j-1] < sizes_x[j-1])
                {
                    break;
                }
                indices[j - 1] = 0;
            }

        }

        //auto grad
        if (x->auto_grad || y->auto_grad) {
            res.auto_grad = true;
            std::shared_ptr<AddNode> node = std::make_shared<AddNode>(res.sizes());

            if (x->auto_grad) node->add_next_edge(x->get_or_create_leaf_node(0));
            if (y->auto_grad) node->add_next_edge(y->get_or_create_leaf_node(1));

            res.grad_fn = node;
        }
        return res;

    }

    Tensor multiply(Tensor* x, Tensor* y)

    {

        //get sizes
        const std::vector<std::size_t>& sizes_x = x->sizes();
        const std::vector<std::size_t>& sizes_y = y->sizes();
        if (sizes_x != sizes_y)
        {

            throw std::runtime_error("matrices must  have the same size for multiplication");
        }
        //create the output tensor
        Tensor res = zeros(sizes_x);
        //get the data and convert it:
        double* data_x = x->data_ptr();
        double* data_y = y->data_ptr();
        double* data_res = res.data_ptr();
        //initiate number of element and number of dimention
        size_t numel = x->numel();
        size_t numd = x->sizes().size();
        //initiate teh indices :
        std::vector<size_t> indices(numd, 0);
        //the actual loop 
        for (size_t i = 0; i < numel; i++)
        {
            size_t pos_x = x->get_flatten_indice(indices);
            size_t pos_y = y->get_flatten_indice(indices);

            data_res[i] = data_x[i] * data_y[i];

            for (size_t j = numd; j > 0; j--)
            {
                indices[j - 1]++;
                if (indices[j - 1] < sizes_x[j - 1])
                {
                    break;
                }
                indices[j - 1] = 0;
            }

        }
      

// Check if autograd is needed
     //auto grad
        if (x->auto_grad || y->auto_grad) {
            res.auto_grad = true;
            std::shared_ptr<MultNode> node = std::make_shared<MultNode>(x,y, res.sizes());

            if (x->auto_grad) node->add_next_edge(x->get_or_create_leaf_node(0));
            if (y->auto_grad) node->add_next_edge(y->get_or_create_leaf_node(1));

            res.grad_fn = node;
        }

        return res;

    }
// produit scalaire entre deux matrices de dimention 2 :

Tensor mm( Tensor* x, Tensor* y)
  {
    const std::vector<std::size_t>& sizes_x = x->sizes();
    const std::vector<std::size_t>& sizes_y = y->sizes();
    //check that the number of dimention is 2
    if (sizes_x.size() != 2 || sizes_y.size()!= 2)
    {

        throw std::runtime_error("matrices must  have the same size for multiplication");
    }
    //check the comptabiliter of dimention
    if (sizes_x[1]!=sizes_y[0])
    {

        throw std::runtime_error("matrices must  have compatible dimentions");
    }
    //create the output tensor
    const std::vector<std::size_t> sizes_out = { sizes_x[0],sizes_y[1] };
    Tensor res = zeros(sizes_out);
    //get the data and convert it:
     double* data_x = x->data_ptr();
     double* data_y = y->data_ptr();
    double* data_res = res.data_ptr();
    // initialize number of rows and cols
    std::size_t rows = sizes_out[0];
    std::size_t cols = sizes_out[1];
    std::size_t dim = sizes_x[1];
    //the loop
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            std::double_t result = 0;
            // on parcours the first row and first cols
            for (size_t s = 0; s < dim; s++)
            {
                 std::vector<std::size_t> indices_x = { i,s };
                 std::vector<std::size_t> indices_y = {s,j };
                 
                size_t pos_x = x->get_flatten_indice(indices_x);
                size_t pos_y = y->get_flatten_indice(indices_y);
                result += data_x[pos_x] * data_y[pos_y];
           
                
            }
            std::vector<std::size_t> indices_out = { i,j };
            size_t pos_out = res.get_flatten_indice(indices_out);
            data_res[pos_out] = result;
        }
    }
    // Check if autograd is needed
     //auto grad
 
    if (x->auto_grad || y->auto_grad) {
       
        res.auto_grad = true;
       std::shared_ptr<mmNode> node = std::make_shared<mmNode>(x, y, res.sizes());
        if (x->auto_grad) node->add_next_edge(x->get_or_create_leaf_node(0));
        if (y->auto_grad) node->add_next_edge(y->get_or_create_leaf_node(1));
  
      
        res.grad_fn = node;
   
    }
    
    return res;

  }

//Transpose a matrice 
Tensor Transpose(Tensor* x)
{
    std::vector < std::size_t > new_sizes = { x->tensorImpl->sizes[1],x->tensorImpl->sizes[0] };
    std::vector < std::size_t > new_strides = { x->tensorImpl->strides[1], x->tensorImpl->strides[0] };
    std::shared_ptr<TensorImpl> new_tensor_impl = std::make_shared<TensorImpl>(x->tensorImpl->storage->clone(),new_sizes,new_strides);
    return Tensor(new_tensor_impl);
}


}
