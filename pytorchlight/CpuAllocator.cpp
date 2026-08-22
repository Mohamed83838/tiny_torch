#include "CpuAllocator.h"
#include <malloc.h> 
#include <new>

void* CpuAllocator::allocate(size_t num_bytes)
{
    if (num_bytes == 0) return nullptr;

    // 64-byte alignment for AVX/SIMD optimizations
    constexpr size_t alignment = 64;

    // Windows-specific aligned allocation
    void* raw_ptr = _aligned_malloc(num_bytes, alignment);

    if (!raw_ptr) {
        throw std::bad_alloc();
    }

    return raw_ptr;
}

void CpuAllocator::deallocate(void* ptr)
{
    if (!ptr) return;

    // Must match _aligned_malloc on Windows
    _aligned_free(ptr);
}
