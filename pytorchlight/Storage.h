#pragma once

#include "Allocator.h"
#include <cstddef>
#include <memory>
#include "Dtype.h"
#include "Device.h"

class CpuAllocator; // Déclaration anticipée requise par le compilateur

class Storage {
private:
    void* data_ptr_;
    std::size_t size_bytes_;
    std::shared_ptr<CpuAllocator> allocator_;
    Dtype dtype_ = Dtype::float64;
    Device device_ = Device::cpu;

public:
    // Lifecycle Constructors
    explicit Storage(std::size_t num_bytes);
    ~Storage();

    // Prohibit structural data duplication via implicit copying
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    //deep copy of storage
    std::shared_ptr<Storage> clone();
    // Allow optimal resource transfer via move semantics
    Storage(Storage&& other) noexcept
        : data_ptr_(other.data_ptr_),
        size_bytes_(other.size_bytes_),
        allocator_(std::move(other.allocator_)),
        dtype_(other.dtype_),
        device_(other.device_) {
        other.data_ptr_ = nullptr;
        other.size_bytes_ = 0;
    }

    Storage& operator=(Storage&& other) noexcept {
        if (this != &other) {
            this->release();

            data_ptr_ = other.data_ptr_;
            size_bytes_ = other.size_bytes_;
            allocator_ = std::move(other.allocator_);
            dtype_ = other.dtype_;
            device_ = other.device_;

            other.data_ptr_ = nullptr;
            other.size_bytes_ = 0;
        }
        return *this;
    }

    // High-Performance Inlined Accessors
    inline void* data() const noexcept { return data_ptr_; }
    inline std::size_t size_bytes() const noexcept { return size_bytes_; }
    inline std::shared_ptr<CpuAllocator> allocator() const noexcept { return allocator_; }

private:
    void release() noexcept;
};