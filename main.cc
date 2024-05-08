#include <vector>
#include <iostream>

//stand-in as a generic activation function
int activation_function(int input) {
	return input;
}

//a single perceptron
class perceptron {
	public:
		perceptron(int count_inputs) {
			for (int i = 0; i < count_inputs; i++) {
				weights.push_back(1); //make this random
			}
		}
		int calculate(const std::vector<int> input) {
			int weighted_sum = 0;
			for (size_t i = 0; i < input.size(); i++) {
				weighted_sum += input[i] * weights[i];
			}
			//
			return activation_function(weighted_sum);
		};
		std::vector<int> weights;
		//how to store activation function? hardcode within calculate() for now?
};

//a layer of perceptrons
class layer {
	public:
		/*
		 * width - the number of neurons in this layer
		 * input_width - number of neurons in previous layer.
		 * 	The accepted number of inputs for this layer.
		 */
		layer(int width, int input_width) {
			this->width = width;
			this->input_width = input_width;

			for (int i = 0; i < width; i++) {
				neurons.push_back(perceptron(input_width));
			}
		};
		std::vector<int> output(std::vector<int> input) {
			std::vector<int> out = {};
			for (size_t i = 0; i < neurons.size(); i++) {
				out.push_back(neurons[i].calculate(input));
			}
			return std::move(out);
		}
		//overload to allow accessing each neuron using [index]?
		std::vector<perceptron> neurons;
		int width; //neurons in layer
		int input_width; //inputs to layer
};

int main(void) {
	//preparations
	const int width = 2;
	const int depth = 4;
	std::vector<int> input = {0,1}; //total inputs must == width
	//generate
	std::vector<layer> layers; //0 = input, last is output
	for (int i = 0; i < depth-1; i++) {
		layers.emplace_back(layer(width,width));
	}
	layers.emplace_back(layer(1, width));
	//output
	std::vector<std::vector<int>> outputs;
	outputs.push_back(layers[0].output(input));
	for (int i = 1; i < depth; i++) {
		outputs.push_back(layers[i].output(outputs[i-1]));
	}
	//interpret
	std::cout << "INPUT: ";
	for (auto x : input)
		std::cout << x << ",";
	std::cout << "\n";
	for (int i = 0; i < depth; i++) { //for-each layer
		std::cout <<  "Layer[" << i << "]->";
		layer *L = &layers[i];
		int layer_width = L->width;
		int input_width = L->input_width;
		//Output vector
		for (int k = 0; k < layer_width; k++) { //print output
			std::cout << outputs[i][k] << ",";
		}
		std::cout << "\n";
		//Neuron weights
		for (int j = 0; j < layer_width; j++) { //for-each neuron
			std::cout << "neuron[" << j << "]: w{";
			for (int k = 0; k < input_width; k++) { //print weights
				std::cout << L->neurons[j].weights[k] << ",";
			}
			std::cout << "}\n";
		}
	}
	return 0;
}
