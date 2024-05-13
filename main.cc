#include <cstdlib>
#include <ctime>
#include <stdexcept>
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
			//passthrough
			return input;
		}
		std::vector<int> weights;
		//how to store activation function? hardcode within calculate() for now?
};
class sign_perceptron : public perceptron {
	public:
		sign_perceptron(int count_inputs) : perceptron(count_inputs) {
			//
		}
		int activation(int input) override {
			//sign activation function
			const int threshold = 0;
			if (input < threshold) return -1;
			if (input == threshold) return 0;
			else return 1;
		}
};
enum activation_type {
	passthrough,
	sign
};
//a layer of perceptrons
class layer {
	public:
		/*
		 * width - the number of neurons in this layer
		 * input_width - number of neurons in previous layer.
		 * 	The accepted number of inputs for this layer.
		 */
		layer(int width, int input_width, activation_type type = sign) {
			this->width = width;
			this->input_width = input_width;

			if (type == passthrough) for (int i = 0; i < width; i++) {
				neurons.emplace_back(new perceptron(input_width));
			}
			else if (type == sign) for (int i = 0; i < width; i++) {
				neurons.emplace_back(new sign_perceptron(input_width));
			}
			else throw std::runtime_error("INVALID ACTIVATION TYPE");
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
		int input_width; //inputs to layer. 0 ifeq input layer.
};

class network {
public:
	network(int width, int depth) {
		this->depth = depth;
		this->width = width;
		//generate
		for (int i = 0; i < depth-1; i++) {
			layers.emplace_back(layer(width,width, sign));
		}
		layers.emplace_back(layer(1, width, passthrough)); //single output node
	}
	//compute output of network given input
	std::vector<int> compute(std::vector<int> input, bool print=true) {
		if (!print) {
			std::vector<int> last_output = layers[0].output(input);
			for (int i = 1; i < depth; i++){
				last_output = layers[i].output(last_output);
			}
			return last_output;
		}
		std::cout << "INPUT: ";
		for (auto x : input)
			std::cout << x << ",";
		std::cout << "\n";
		std::vector<std::vector<int>> outputs;
		outputs.push_back(layers[0].output(input));
		for (int i = 1; i < depth; i++) {
			outputs.push_back(layers[i].output(outputs[i-1]));
		}
		//interpret
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
		return std::move(outputs[depth]);
	}
	void measure(std::vector<int> input, std::vector<int> target) {
		//process input
		std::vector<std::vector<int>> outputs
			= { layers[0].output(input) };
		for (int i = 1; i < depth; i++) {
			outputs.emplace_back(layers[i].output(outputs[i-1]));
		}
		//display
		std::cout << "Target:\t";
		for (auto& t : target) std::cout << t << ",";
		std::cout << "\nIn:\t";
		for (auto& i : input) std::cout << i << ",";
		for (int j = 0; j < depth; j++) {
			if (j+1 == depth)
				std::cout << "\nOut:\t";
			else
				std::cout << "\n" << j << ":\t";
			for (auto& o : outputs[j]) std::cout << o << ",";
		}
		//calc error & display
		std::cout << "\nERROR = ";
		for (int x = 0; x < layers[depth-1].width; x++) {
			std::cout << target[x]-outputs[depth-1][x] << ",";
		}
		std::cout << "\n";

	}
private:
	int width,depth;
	std::vector<layer> layers; //0 = first-layer, last is output; input 'layer' is just the vector given
};

int main(void) {
	std::srand(time(NULL));
	//preparations
	const int width = 2;
	const int depth = 3;
	network n(width,depth); //the network
	n.compute({0,1});	 //total inputs must == width
	n.measure({0,1},{2});
	return 0;
}
