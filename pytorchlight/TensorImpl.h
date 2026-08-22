#pragma once
#include "Storage.h"
#include <cstddef>
#include <vector>
#include <memory> 
class TensorImpl {

public:
	std::vector<size_t> sizes;
	std::vector<size_t> strides;
	std::size_t offset=0;
	std::size_t dim;
	std::size_t size_in_bytes;
	std::shared_ptr<Storage> storage;


		TensorImpl(std::vector<size_t> sizes_);
		TensorImpl(std::shared_ptr<Storage> storage, std::vector<size_t> sizes_, std::vector<size_t> strides_ );
		TensorImpl(std::shared_ptr<Storage> storage, std::vector<size_t> sizes_);
		~TensorImpl()=default;

		std::size_t get_flat_index(const std::vector<std::size_t>& indices);
		std::size_t numel();
		std::shared_ptr<TensorImpl> clone();

	private:
		void Compute_default_strides();


	



};