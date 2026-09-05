#include "Auto_grad.h"
#include <iostream>

namespace tiny_torch {
	namespace auto_grad {
		//hello function to test the namespace
		std::string hello()
		{
			return "hello from auto_grad namespace";
		}

		//create Node tasks for each node in the graph, starting from the root node
		std::vector<std::shared_ptr<NodeTask>> create_node_tasks(std::shared_ptr<Node> node, bool is_root)
		{
		
			if (node->is_leaf_node)
			{
				return {};
			}
			
			std::vector<std::shared_ptr<NodeTask>> node_tasks;
			if (is_root)
			{
				
				auto new_node = std::make_shared<NodeTask>(node, 0, std::vector<std::shared_ptr <NodeTask>>{});
				node->node_task = new_node;
				node_tasks.push_back(new_node);
			}

			for (std::shared_ptr<Edge> edge : node->next_edges_)
			{
				//check if NodeTask exist than update the dependencies of the NodeTask
				if (edge->next_function->node_task == nullptr)
				{

					auto new_node = std::make_shared<NodeTask>(edge->next_function, 1, std::vector<std::shared_ptr <NodeTask>>{ node->node_task });
					edge->next_function->node_task = new_node;
					node_tasks.push_back(new_node);
					//recursively create NodeTasks for the next function and its children
					std::vector<std::shared_ptr <NodeTask>> child_node_tasks = create_node_tasks(edge->next_function, false);
					node_tasks.insert(
						node_tasks.end(),
						child_node_tasks.begin(),
						child_node_tasks.end()
					);
				}
				else
				{
					edge->next_function->node_task->nb_dependencies++;
					edge->next_function->node_task->dependencies.push_back(node->node_task);

				}
		
			}

			return node_tasks;
		}

		//backward function to start the backward pass from the root node
		void backward(std::shared_ptr<Node> rootNode)
		{
			//create NodeTasks for each node in the graph starting from the root node
			std::vector<std::shared_ptr<NodeTask>> que = create_node_tasks(rootNode, true);

			//grad for the root node is a tensor of ones with the same size as the root node's tensor
			Tensor grad_root = tiny_torch::fill(rootNode->size_of_tensor, 1);
			rootNode->node_task->grad_to_execute = std::make_shared<Tensor>(std::move(grad_root));

			std::vector<std::shared_ptr<NodeTask>> ready_tasks;
			while (!que.empty())
			{

				// 1. Collect ready tasks
				for (const std::shared_ptr<NodeTask>& task : que)
				{
					if (task->nb_dependencies == 0)
					{
						ready_tasks.push_back(task);
					}
				}

				// 2. Erase ready tasks from que in O(N) using Erase-Remove
				que.erase(
					std::remove_if(que.begin(), que.end(), [](const std::shared_ptr<NodeTask>& task) {
						return task->nb_dependencies == 0;
						}),
					que.end()
				);

				

				// 3. Execute ready tasks
				for (const std::shared_ptr<NodeTask>& task : ready_tasks)
				{
					
					std::vector<Tensor> grad_to_execute = { *task->grad_to_execute };

					std::vector<Tensor> grads = task->node->backward(grad_to_execute);
					
					if (!task->node->is_leaf_node)
					{
						
						//distribute the gradients to the dependencies of the task
						for (const std::shared_ptr<Edge>& edge : task->node->next_edges_)
						{
							//update the grad_to_execute of the dependency NodeTask

							if (edge->next_function->node_task->grad_to_execute == nullptr)
							{
								edge->next_function->node_task->grad_to_execute = std::make_shared<Tensor>(std::move(grads[edge->input_nr]));
							}
							else
							{
								*edge->next_function->node_task->grad_to_execute = tiny_torch::add(edge->next_function->node_task->grad_to_execute.get(), &grads[edge->input_nr]);
							}


							edge->next_function->node_task->dependencies.erase(
								std::remove(edge->next_function->node_task->dependencies.begin(),
									edge->next_function->node_task->dependencies.end(),
									task),
								edge->next_function->node_task->dependencies.end()
							);
							edge->next_function->node_task->nb_dependencies--;
						}
					}
				}

				// 4. Reset ready_tasks for the next loop
				ready_tasks.clear();
			}
		}

	
		
	}
}