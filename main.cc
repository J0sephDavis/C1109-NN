#include <cstdlib>
#include <ctime>
#include <vector>
#include <iostream>
#include <memory> //unique_ptr

//a single perceptron
class perceptron {
	public:
		perceptron(int count_inputs) {
			for (int i = 0; i < count_inputs; i++) {
				weights.push_back((std::rand()% 10) - 4);
			}
		}
		int calculate(const std::vector<int> input) {
			int weighted_sum = 0;
			for (size_t i = 0; i < input.size(); i++) {
				weighted_sum += input[i] * weights[i];
			}
			return activation(weighted_sum);
		};
		virtual int activation(int input) {
			//sign activation function
			const int threshold = 0;
			if (input < threshold) return -1;
			if (input == threshold) return 0;
			else return 1;

		}
		std::vector<int> weights;
		//how to store activation function? hardcode within calculate() for now?
};
class output_perceptron : public perceptron {
	public:
		output_perceptron(int count_inputs) : perceptron(count_inputs) {
			//
		}
		int activation(int input) override {
			return input;
		}
};

//a layer of perceptrons
class layer {
	public:
		/*
		 * width - the number of neurons in this layer
		 * input_width - number of neurons in previous layer.
		 * 	The accepted number of inputs for this layer.
		 */
		layer(int width, int input_width, bool is_output) {
			this->width = width;
			this->input_width = input_width;
			
			if (is_output) for (int i = 0; i < width; i++) {
				std::cout << "create output_perceptron\n";
				neurons.emplace_back(new output_perceptron(input_width));
			}
			else for (int i = 0; i < width; i++) {
				neurons.emplace_back(new perceptron(input_width));
			}
		};
		std::vector<int> output(std::vector<int> input) {
			std::vector<int> out = {};
			for (size_t i = 0; i < neurons.size(); i++) {
				out.push_back(neurons[i]->calculate(input));
			}
			return std::move(out);
		}
		//overload to allow accessing each neuron using [index]?
		std::vector<std::unique_ptr<perceptron>> neurons;
		int width; //neurons in layer
		int input_width; //inputs to layer
};

int main(void) {
	std::srand(time(NULL));
	//preparations
	const int width = 2;
	const int depth = 4;
	std::vector<int> input = {0,1}; //total inputs must == width
	//generate
	std::vector<layer> layers; //0 = input, last is output
	for (int i = 0; i < depth-1; i++) {
		layers.emplace_back(layer(width,width, false));
	}
	layers.emplace_back(layer(1, width,true)); //single output node
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
				std::cout << L->neurons[j]->weights[k] << ",";
			}
			std::cout << "}\n";
		}
	}
	return 0;
}
