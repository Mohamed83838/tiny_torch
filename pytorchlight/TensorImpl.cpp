#include "TensorImpl.h"
#include <cstddef>
#include <memory>
#include "Storage.h"
#include <iostream>

TensorImpl::TensorImpl(std::vector<size_t> sizes_)
	:sizes(sizes_)
{
	size_t res = numel();
	size_in_bytes = res * sizeof(double);
	storage = std::make_shared<Storage>(res);
	Compute_default_strides();

}
TensorImpl::TensorImpl(std::shared_ptr<Storage> storage_,std::vector<size_t> sizes_)
	:sizes(sizes_),storage(storage_)
{
	size_t res = numel();
	size_in_bytes = res * sizeof(double);
	Compute_default_strides();
}
TensorImpl::TensorImpl(std::shared_ptr<Storage> storage_, std::vector<size_t> sizes_, std::vector<size_t> strides_)
	:sizes(sizes_), storage(storage_),strides(strides_)
{
	size_t res = numel();
	size_in_bytes = res * sizeof(double);
}

std:: size_t TensorImpl::numel()
{
	dim = sizes.size();
	std::size_t res = 1;
	for (size_t i = 0; i < dim; i++)
	{
		res *= sizes[i];
	};
	return res;
}


std::size_t TensorImpl::get_flat_index(const std::vector<size_t>& indices) 
{
	if (indices.size() != sizes.size()) {
		throw std::runtime_error("Index dimensions do not match tensor dimensions.");
	}
	size_t res = offset;
	size_t len = indices.size();
	for (size_t i = 0; i <len ; i++)
	{
		if (indices[i] < 0 || indices[i] >= sizes[i]) {
			throw std::out_of_range("Index out of bounds.");
		}
		res += indices[i] * strides[i];
	}
	return res;
}

void TensorImpl::Compute_default_strides()
{
	std::size_t len = sizes.size();
	if (len == 0) return;

	strides.resize(len);
	std::size_t running_stride = 1;
	for (size_t i = len; i > 0; --i)
	{
		strides[i - 1] = running_stride;
		running_stride *= sizes[i - 1];
	}
}

std::shared_ptr<TensorImpl> TensorImpl::clone()
{
	std::shared_ptr<Storage> new_storage = storage->clone();
	return std::make_shared<TensorImpl>(new_storage, sizes, strides);
}