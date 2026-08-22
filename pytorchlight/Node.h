#pragma once
#include <vector> 
#include "memory"
#include <cstddef> 
#include <atomic>
#include <cstdint>
#include "Tensor.h"
#include "Edge.h"
class Node
{
public:
	std::vector<std::shared_ptr<Edge>> next_edges_;
	std::size_t sequence_nr;
	Node();
	virtual ~Node() = default;
	virtual std::vector<Tensor> backward(const std::vector<Tensor>& grads) ;
	void add_next_edge(std::shared_ptr<Edge> next_edge);

private:
	static uint64_t get_next_sequence_nr() {
		static std::atomic<uint64_t> global_seq{ 0 };
		return ++global_seq; // Thread-safe atomic increment
	}

};