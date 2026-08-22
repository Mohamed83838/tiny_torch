#pragma once

#include <memory>
class Node;

class Edge
{
public:
	int input_nr=0;
	std::shared_ptr<Node> next_function=nullptr;
	Edge(int input_nr, std::shared_ptr<Node> next_function);

};