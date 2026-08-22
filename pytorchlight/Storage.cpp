#include "Storage.h"
#include "CpuAllocator.h" // Full definition needed here to call member functions
#include <cstddef>
#include <utility>
#include <cstring>

Storage::Storage(std::size_t numel)
    : data_ptr_(nullptr), size_bytes_(numel*sizeof(double)) {
    allocator_ = std::make_shared<CpuAllocator>();
    if (allocator_) {
        data_ptr_ = allocator_->allocate(size_bytes_);
    }
}

Storage::~Storage() {
    release();
}

void Storage::release() noexcept {
    if (data_ptr_ && allocator_) {
        allocator_->deallocate(data_ptr_);
        data_ptr_ = nullptr;
    }
}

std::shared_ptr<Storage> Storage::clone() 
{
    std:: shared_ptr<Storage> new_storage = std::make_shared<Storage>(size_bytes_ / sizeof(double));
    if (data_ptr_ && new_storage->data())
    {
        std::memcpy(new_storage->data(), data_ptr_, size_bytes_);
    }
    return new_storage;
}