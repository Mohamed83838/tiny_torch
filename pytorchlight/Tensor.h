#pragma once
#include "TensorImpl.h"
#include <string>
#include "Dtype.h"
#include<memory>
#include "Edge.h"
class Node;

class Tensor {


public:
	//autograd params
	bool auto_grad = false;
	std::shared_ptr<Node> grad_fn = nullptr;
	
	std::shared_ptr <TensorImpl> tensorImpl;
	Tensor(std::vector<std::size_t> sizes_);
	Tensor(std::shared_ptr<TensorImpl> impl);
	~Tensor() = default;
	//autograd tensor
	std::shared_ptr <Tensor> grad = nullptr;
	
	const std::vector<size_t>& sizes() const { return tensorImpl->sizes; }
	const std::vector<size_t>& strides() const { return tensorImpl->strides; }
	size_t numel() const { return tensorImpl->numel(); }
	void write(const std::vector<size_t>& indices, double value);
	double read(const std::vector<size_t>& indices);
	std::size_t get_flatten_indice(const std::vector<size_t>& indices);
	std::string get_elements(std::vector<size_t>& indices);
	double* data_ptr();
	std::string lineariser(std::size_t depth, std::vector<size_t> indices);
	std::string print();
	std::shared_ptr<Tensor> grad_tensor();
	void transpose();

	std::shared_ptr<Edge> get_or_create_leaf_node(int input_nr_);
	std::shared_ptr<Tensor> clone();
};