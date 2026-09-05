#pragma once
#include <string>
#include <memory>
#include <vector>
#include "Node.h"
#include "TensorFactories.h"
namespace tiny_torch {
	namespace auto_grad
	{
		
		//void backward(std::shared_ptr<Node> rootNode);
		std::vector<std::shared_ptr<NodeTask>> create_node_tasks(std::shared_ptr<Node> node,bool is_root);
		void backward(std::shared_ptr<Node> rootNode);
		
	

		
	}
}