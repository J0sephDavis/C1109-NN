#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <memory> //unique_ptr
#include <math.h>

//a single perceptron
class perceptron {
	public:
		perceptron(int count_inputs) {
			for (int i = 0; i < count_inputs; i++) {
				weights.push_back((std::rand()% 4 + 1));
			}
		}
		float calculate(const std::vector<float> input) {
			for (size_t i = 0; i < input.size(); i++) {
				weighted_sum += input[i] * weights[i];
			}
			return activation(weighted_sum);
		};
		virtual float activation(float input) {
			//logistic sigmoid
			output = 1/(1+exp(-input)); //possible underflow
			return output;
		}
		std::vector<float> weights;
		float weighted_sum = 0;
		float error_contribution = 0;
		float output = 0.0; //the last output of this node.
};
class sign_perceptron : public perceptron {
	public:
		sign_perceptron(int count_inputs) : perceptron(count_inputs) {
			//
		}
		float activation(float input) override {
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
				//neurons.emplace_back(new sign_perceptron(input_width));
				neurons.emplace_back(new perceptron(input_width));
			}
			else throw std::runtime_error("INVALID ACTIVATION TYPE");
		};
		std::vector<float> output(std::vector<float> input) {
			std::vector<float> out = {};
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
	network(int input_width, int width, int depth) {
		this->depth = depth;
		this->width = width;
		//generate
		layers.emplace_back(layer(width,input_width, sign));
		for (int i = 1; i < depth-1; i++) {
			layers.emplace_back(layer(width,width, sign));
		}
		layers.emplace_back(layer(1, width, passthrough)); //single output node
	}
	//compute output of network given input
	std::vector<std::vector<float>> compute(std::vector<float> input) {
		std::cout << "INPUT: ";
		for (auto x : input)
			std::cout << x << ",";
		std::cout << "\n";
		std::vector<std::vector<float>> outputs;
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
		return std::move(outputs);
	}
	int width,depth;
	std::vector<layer> layers; //0 = first-layer, last is output; input 'layer' is just the vector given
};

int main(void) {
	//preparations
	std::srand(time(NULL));
	const int width = 6;
	const int depth = 3;
	const std::vector<float> inputs = {1,1};
	const std::vector<float> expected_output = {1.0};
	const float learning_rate = 0.25;
	network n(inputs.size(), width,depth); //the network
	std::vector<std::vector<float>> output = n.compute(inputs);

	std::cout << "compute gradient\n";
	//get error contribution of output nodes
	for (size_t i = 0; i < output.back().size(); i++){
		std::cout << "==\n";
		auto& neuron = n.layers.back().neurons.at(i);
		float error_signal = expected_output.at(i) - neuron->output;
		float derivative = (neuron->output * (1-neuron->output));
		float error_contribution = error_signal * derivative;
		std::cout << "output error/contrib: " << error_signal << "/"
			<< error_contribution << "\n";
		//d_pj = (t_pj - o_pj) * o_pi * (1-o_pj)
		neuron->error_contribution = error_contribution;
		//DELTA RULE
		for (size_t weight_index = 0;
				weight_index < neuron->weights.size()-1;
				weight_index++) {
			std::cout << "w_" << weight_index << "\t"
				<< neuron->weights.at(weight_index) << "\n";
			auto& input_neuron = n.layers.at(n.layers.size()-2).neurons.at(weight_index);
			float delta_rule = -learning_rate * error_contribution * input_neuron->output;
			std::cout << "delta W\t"
				<< delta_rule << "\n";
			neuron->weights.at(weight_index) = neuron->weights.at(weight_index) + delta_rule;

		}
	}
	for (int i = n.layers.size()-2; i >= 0; i--) {
		std::cout << "==\nLayer-" << i << "\n";
		auto& layer = n.layers.at(i);
		auto& upper_layer = n.layers.at(i+1);
		std::vector<float> input_values = {};
		if (i == 0) {
			input_values = inputs;
		}
		else {
			for (size_t a = 0; a < n.layers.at(i-1).neurons.size()-1; a++) {
				auto& tmp = n.layers.at(i-1).neurons.at(a);
				input_values.push_back(tmp->output);
			}
		}
		std::cout << "layer input:";
		for (auto& inval : input_values) {
			std::cout << inval << ",";
		}
		std::cout <<"\n";
		for (size_t j = 0; j < layer.neurons.size(); j++) {
			//d_pk = f_j'(z_pj) * SUM_k (d_pk * w_kj)
			//where the k-neurons are an upper layer
			std::cout << ">Neuron-" << j << "\n";
			auto& neuron = layer.neurons.at(j);
			float derivative = neuron->output * (1 - neuron->output);
			float associated_error = 0;
			for (size_t k = 0; k < upper_layer.neurons.size(); k++) { //nodes this feeds into
				auto& uppper_neuron = upper_layer.neurons.at(k);
				//d_pk * w_kj
				float upper_contribution = uppper_neuron->error_contribution
					* uppper_neuron->weights.at(j);
				associated_error += upper_contribution;
			}
			std::cout << "sum_k:\t" << associated_error << "\n";
			neuron->error_contribution = associated_error * derivative; //d_pk
			std::cout << "d_pk:\t" << neuron->error_contribution << "\n";
			std::cout << "WEIGHT COUNT:" << neuron->weights.size() << "\n";
			for (size_t l = 0; l <= neuron->weights.size()-1;l++) {

				float input_value = input_values[l];
				float delta_rule = -learning_rate * neuron->error_contribution *
					input_value;
				std::cout << "delta(w_" << l << "):\t" << delta_rule << "\n";
				neuron->weights[l] = delta_rule + neuron->weights[l];
			}
		}
	}
	return 0;
}
