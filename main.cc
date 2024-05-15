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
		std::cout << "INPUT\t";
		for (auto x : input)
			std::cout << x << ", ";
		std::cout << "\n";
		std::vector<std::vector<float>> outputs;
		outputs.push_back(layers[0].output(input));
		for (int i = 1; i < depth; i++) {
			outputs.push_back(layers[i].output(outputs[i-1]));
		}
		//interpret
		for (int i = 0; i < depth; i++) { //for-each layer
			std::cout <<  "#L-" << i << " OUTPUT\t";
			layer *L = &layers[i];
			int layer_width = L->width;
			int input_width = L->input_width;
			//Output vector
			for (int k = 0; k < layer_width; k++) { //print output
				std::cout << outputs[i][k] << ", ";
			}
			std::cout << "]\n";
			//Neuron weights
			for (int j = 0; j < layer_width; j++) { //for-each neuron
				std::cout << "*N-" << j << " w[]  ";
				for (int k = 0; k < input_width; k++) { //print weights
					std::cout << L->neurons[j]->weights[k] << ", ";
				}
				std::cout << "\n";
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
	const int width = 2;
	const int depth = 2;
	const std::vector<float> inputs = {1,0.3};
	const std::vector<float> expected_output = {0.4};
	const float learning_rate = 0.25;
	network n(inputs.size(), width,depth); //the network
	std::vector<std::vector<float>> output = n.compute(inputs);

	//get error contribution of output nodes
	for (size_t output_index = 0; output_index < output.back().size();
			output_index++){
		std::cout << "\n#L-" << output.size()-1 << "OUTPUT GRADIENT\n";
		auto& neuron = n.layers.back().neurons.at(output_index);
		float error_strength = expected_output.at(output_index) - neuron->output;
		float derivative = (neuron->output * (1-neuron->output));

		neuron->error_contribution = error_strength * derivative;
		std::cout << "(target - actual output)\t|" << error_strength;
		std::cout << "\n\t\t\tf'(net)\t| " <<  derivative;
		std::cout << "\n\t\td=(t-o)f'(net)\t|" << neuron->error_contribution << "\n";
		if (error_strength == 0) {
			std::cout << "no error. Stop.\n";
			return 0;
		}
		//d_pj = (t_pj - o_pj) * f'(z_pj)
		//t_pj - target output
		//o_pj - actual output (in this case of the output node)
		//z_pj - weighted sum of inputs
		//derivative of logistic = f(x)(1-f(x))
		
		//DELTA RULE - the change in weight (from neuron u_i to u_j) is
		//  equal to -1 * learning rate, multiplied by
		//  error contribution (d_pj) then multiplied by the output of
		//  neuron-i (u_i)
		for (size_t weight_index = 0;
				weight_index < neuron->weights.size()-1;
				weight_index++) {
			std::cout << "\t\t\tW-" << weight_index << "\t|"
				<< neuron->weights.at(weight_index) << "\n";
			auto& input_neuron = n.layers.at(n.layers.size()-2)
				.neurons.at(weight_index);
			float delta_weight = -learning_rate * 	//-learning rate
				neuron->error_contribution * 	//d_pj
				input_neuron->output; 		//o_pi
			std::cout << "\t\t\tdelta\t|" << delta_weight << "\n";
			neuron->weights.at(weight_index) = neuron->weights
				.at(weight_index) + delta_weight; // w + dW

		}
	}
	//hidden layers
	for (int layer_index = n.layers.size()-2; layer_index >= 0;
			layer_index--) {
		std::cout << "\n#L-" << layer_index;

		auto& layer = n.layers.at(layer_index);
		auto& upper_layer = n.layers.at(layer_index+1);
		std::vector<float> input_values = {}; //

		if (layer_index == 0) //INPUT layer
			input_values = inputs; //Grab the test-set input
		else for (size_t input_k = 0; input_k < n.layers
				.at(layer_index-1).neurons.size()-1;
				input_k++) {
			auto& tmp = n.layers.at(layer_index-1).neurons.at(input_k);
			input_values.emplace_back(tmp->output);
		}
		std::cout << " INPUT\t";
		int tmp = 0;
		for (auto& inval : input_values) {
			std::cout << inval << ", ";
		}
		std::cout <<"\n";
		for (size_t u_j = 0; u_j < layer.neurons.size(); u_j++) {
			//d_pj = f_j'(z_pj) * SUM_k (d_pk * w_kj)
			//where the k-neurons are an upper layer
			std::cout << "*N-" << u_j << "\n";
			auto& neuron = layer.neurons.at(u_j);
			float derivative = neuron->output * (1 - neuron->output);
			std::cout << "f_j'(z_pj)\t|" << derivative << "\n";
			float associated_error = 0; // SUM_k (d_pk * w_kj)
			for (size_t k = 0; k < upper_layer.neurons.size(); k++) {
				auto& uppper_neuron = upper_layer.neurons.at(k);
				//d_pk * w_kj
				associated_error += uppper_neuron
					->error_contribution * uppper_neuron
					->weights.at(u_j);
			}
			std::cout << "sum_k\t\t|" << associated_error << "\n";
			neuron->error_contribution = associated_error * derivative; //d_pk
			std::cout << "d_pk:\t\t|" << neuron->error_contribution
				<< "\n";
			std::cout << "delta:\t\t|";
			for (size_t u_i = 0; u_i <= neuron->weights.size()-1;
					u_i++) {
				float input_value = input_values[u_i]; //o_pi
				//delta = -learning_rate * d_pj * o_pi
				float delta_rule = -learning_rate *
					neuron->error_contribution *
					input_value;
				std::cout << delta_rule << ", ";
				neuron->weights[u_j] = delta_rule + neuron->weights[u_j];
			}
			std::cout << "\n";
		}
	}
	return 0;
}
