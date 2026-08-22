#pragma once
#include <cstddef>

class Allocator 
{
	
public :
	virtual ~Allocator() = default;
	virtual void* allocate(std::size_t size) = 0;
	virtual void deallocate(void* ptr) = 0;

};

