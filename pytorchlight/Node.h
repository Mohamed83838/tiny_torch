#pragma once
#include <vector> 
#include "memory"
#include <cstddef> 
#include <atomic>
#include <cstdint>
#include "Tensor.h"
#include "Edge.h"



//Node Task struct
struct NodeTask
{
	std::shared_ptr<Node> node;
	std::size_t nb_dependencies;
	std::vector < std::shared_ptr <NodeTask> >dependencies;
	std::shared_ptr <Tensor> grad_to_execute = nullptr;

	NodeTask(std::shared_ptr<Node> node_,
		std::size_t nb_dependencies_, std::vector < std::shared_ptr <NodeTask> >dependencies_) :
		node(node_), nb_dependencies(nb_dependencies_), dependencies(dependencies_)
	{
	}
};


class Node
{
public:
	std::string node_type = "Node";
	std::vector<std::shared_ptr<Edge>> next_edges_;
	bool is_leaf_node = false;
	std::size_t sequence_nr;
	std::shared_ptr<NodeTask> node_task = nullptr;
	std::vector<size_t> size_of_tensor;
	Node(std::vector<size_t> size_of_tensor_);
	virtual ~Node() = default;
	virtual std::vector<Tensor> backward( std::vector<Tensor>& grads) ;
	void add_next_edge(std::shared_ptr<Edge> next_edge);
	std::string print();
private:
	static uint64_t get_next_sequence_nr() {
		static std::atomic<uint64_t> global_seq{ 0 };
		return ++global_seq; // Thread-safe atomic increment
	}

};

