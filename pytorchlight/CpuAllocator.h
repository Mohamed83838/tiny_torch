#pragma once

#include "Allocator.h"
#include <cstddef>

class CpuAllocator : public Allocator
{
public:
    virtual ~CpuAllocator() override = default;

    virtual void* allocate(size_t num_bytes) override;
    virtual void deallocate(void* ptr) override;
    
};