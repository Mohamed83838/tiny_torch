#include "Tensor.h"
#include <iostream>
#include <string>
#include <format>
#include "AccumulatedGrad.h"


Tensor::Tensor(std::vector<std::size_t>sizes_)
{
	tensorImpl = std::make_shared<TensorImpl>(sizes_);
}

Tensor::Tensor(std::shared_ptr<TensorImpl> impl) :
	tensorImpl(impl) {
};
std::size_t Tensor::get_flatten_indice(const std::vector<size_t>& indices)
{
	return tensorImpl->get_flat_index(indices);
}

void Tensor::write(const std::vector<size_t>& indices, double value)
{
	std::size_t flat_index = tensorImpl->get_flat_index(indices);
	double* size_t_buffer = static_cast<double*>(tensorImpl->storage->data());

	size_t_buffer[flat_index] = value;

}
double Tensor::read(const std::vector<size_t>& indices)
{
	size_t flat_index = tensorImpl->get_flat_index(indices);
	double* size_t_buffer = static_cast<double*>(tensorImpl->storage->data());
	return size_t_buffer[flat_index];

}

double* Tensor::data_ptr()
{
	void* data= tensorImpl->storage->data();
	return  static_cast<double*>(data);
}

std::string Tensor::print()
{
	
	std::vector<size_t> indices(sizes().size(),0);
	if (indices.size() == 0) return "{}";
	return lineariser(0, indices);
}

//recursive function
std::string Tensor::lineariser(std::size_t depth, std::vector<size_t> indices)
{
	const std::vector<size_t>& l_sizes = sizes();
	if (depth == l_sizes.size() -1)
	{
		
		return get_elements(indices);
	}
	std::string res = "{";
	for (size_t i = 0; i < l_sizes[depth]; i++)
	{
		std::vector<size_t> new_indices(indices);
		new_indices[depth] = i;
		if (i > 0) res += ", ";
		res += lineariser(depth + 1, new_indices);
	}
	
	return res +"}\n";
	
}
//helper function to print the data
std::string Tensor:: get_elements(std::vector<size_t>& indices)
{
	double* data = data_ptr();
	const std::vector<size_t>& l_sizes = sizes();
	std::size_t numdim = l_sizes.size();
	std::string res = "{";

	for (std::size_t i = 0; i < l_sizes[numdim - 1]; i++)
	{
		indices[numdim-1] = i;
		if (i > 0) res += ", ";
		std::size_t flat_index = get_flatten_indice(indices);
		res += std::to_string(data[flat_index]) ;

	}
	return res +"}\n";
}

//get grad_tensor
std::shared_ptr<Tensor> Tensor::grad_tensor ()
{
	return grad;
}

//transpose the matrice

void Tensor::transpose()
{
	//for 2 dimentional matrices inverting sizes
	tensorImpl->sizes = { tensorImpl->sizes[1] ,tensorImpl->sizes[0] };
	//changing strides
	tensorImpl->strides = { tensorImpl->strides[1] ,tensorImpl->strides[0] };
	
}

std::shared_ptr <Edge> Tensor::get_or_create_leaf_node(int input_nr_) {
	if (!auto_grad) return nullptr;
	if (grad_fn != nullptr) return std::make_shared<Edge>(input_nr_, grad_fn);

	//grad_fn = std::make_shared<AccumulatedGrad>(*this);
	AccumulatedGrad* p = new AccumulatedGrad(this,this->sizes());
	grad_fn = std::shared_ptr<Node>(p);
	return std::make_shared<Edge>(input_nr_,grad_fn);
}
//clone a tensor
std::shared_ptr<Tensor> Tensor::clone() {
	return std::make_shared<Tensor>(tensorImpl->clone());
}