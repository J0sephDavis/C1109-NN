#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <memory> //unique_ptr
#include <math.h>

//a single perceptron - logistic
class perceptron {
	public:
		perceptron(int count_inputs) {
			for (int i = 0; i < count_inputs; i++) {
				weights.push_back((std::rand()% 4 + 1)*0.25);
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

			//f(x) = 1/1+e^-x
			output = 1/(1+exp(-input)); //possible underflow
			//f'(x) = f(x)(1-f(x);
			derivative = output * (1 - output);
			return output;
		}
		//given an array of inputs, determines the weight changes needed
		virtual void train(float learning_rate, std::vector<float> input) {
			//DELTA RULE - the change in weight (from neuron u_i to u_j) is
			//  equal to -1 * learning rate, multiplied by
			//  error contribution (d_pj) then multiplied by the output of
			//  neuron-i (u_i)
			for (size_t weight_index = 0;
					weight_index < weights.size()-1;
					weight_index++) {
				float delta_weight = -learning_rate *
					error_contribution *
					input[weight_index];
				weights.at(weight_index) = 
					weights.at(weight_index) + delta_weight;
			}
		}
		std::vector<float> weights;
		float weighted_sum = 0;
		float error_contribution = 0;
		float output = 0.0; //the last output of this node.
		float derivative = 0;
};
enum activation_type {
	logistic,
};
//a layer of perceptrons
class layer {
public:
	/*
	 * width - the number of neurons in this layer
	 * input_width - number of neurons in previous layer.
	 * 	The accepted number of inputs for this layer.
	 */
	layer(int width, int input_width, activation_type type = logistic) {
		this->width = width;
		this->input_width = input_width;

		if (type == logistic) for (int i = 0; i < width; i++) {
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
	virtual void train(std::vector<float> input, std::vector<float> label,
			const float learning_rate,
			std::shared_ptr<layer> upper_layer) {
		(void)label; //unused variable warning
		//std::cout << "\n#train" << "[HIDDEN]\n";
		for (size_t u_j = 0; u_j < neurons.size(); u_j++) {
			//d_pj = f_j'(z_pj) * SUM_k (d_pk * w_kj)
			//where the k-neurons are an upper layer
			//std::cout << "*N-" << u_j << "\n";
			auto& neuron = neurons.at(u_j);
			float derivative = neuron->output * (1 - neuron->output);
			//std::cout << "f_j'(z_pj)\t|" << derivative << "\n";
		//==Caclulate 2nd half of error equation - SUM_k (d_pk * w_kj==
			float associated_error = 0;
			for (size_t k = 0; k < upper_layer->neurons.size(); k++) {
				auto& uppper_neuron = upper_layer->neurons.at(k);
				//d_pk * w_kj
				associated_error += uppper_neuron
					->error_contribution * uppper_neuron
					->weights.at(u_j);
			}
			//std::cout << "sum_k\t\t|" << associated_error << "\n";
			neuron->error_contribution = associated_error * derivative; //d_pk
			//std::cout << "d_pk:\t\t|" << neuron->error_contribution
			//	<< "\n";
			//std::cout << "delta:\t\t|";
			neuron->train(learning_rate, (input));
		}
	}
	//overload to allow accessing each neuron using [index]?
	std::vector<std::unique_ptr<perceptron>> neurons;
	int width; //neurons in layer
	int input_width; //inputs to layer. 0 ifeq input layer.
};
class output_layer : public layer {
public:
	output_layer(int width, int input_width, activation_type type = logistic)
		: layer(width,input_width,type){
		//
	}
	void train(std::vector<float> input, std::vector<float> label,
			const float learning_rate,
			std::shared_ptr<layer> upper_layer) override {
		(void)upper_layer;
#ifdef PRINT_TRAIN
		std::cout << "\n#train" << "[OUTPUT]\n";
#endif
		//get error contribution of output nodes
		int label_index = 0;
		for (auto& neuron : neurons) {
			float error_strength
				=label.at(label_index++) - neuron->output;
			float derivative
				= neuron->output * (1 - neuron->output);
			neuron->error_contribution = error_strength * derivative;
			
			//d_pj = (t_pj - o_pj) * f'(z_pj)
			//t_pj - target output
			//o_pj - actual output (in this case of the output node)
			//z_pj - weighted sum of inputs
			//derivative of logistic = f(x)(1-f(x))
#ifdef PRINT_TRAIN
			std::cout << "(target - actual output)\t|"
				<< error_strength << "\n\t\t\tf'(net)\t| "
				<<  derivative << "\n\t\td=(t-o)f'(net)\t|"
				<< neuron->error_contribution << "\n";
#endif
			neuron->train(learning_rate, std::move(input));
		}
	}
};
class network {
public:
	network(int input_width, int width, int depth) {
		this->depth = depth;
		this->width = width;
		//generate
		layers.emplace_back(new layer(width,input_width, logistic));
		for (int i = 1; i < depth-1; i++) {
			layers.emplace_back(new layer(width,width, logistic));
		}
		layers.emplace_back(new output_layer(1, width, logistic)); //single output node
	}
	//compute output of network given input
	std::vector<std::vector<float>> compute(std::vector<float> input) {
		//std::cout << "INPUT\t";
		//for (auto x : input)
			//std::cout << x << ", ";
		//std::cout << "\n";
		std::vector<std::vector<float>> outputs;
		outputs.push_back(layers[0]->output(input));
		for (int i = 1; i < depth; i++) {
			outputs.push_back(layers[i]->output(outputs[i-1]));
		}
		//interpret
		for (int i = 0; i < depth; i++) { //for-each layer
			//std::cout <<  "#L-" << i << " OUTPUT\t";
			std::shared_ptr<layer> layer = layers.at(i);
			int layer_width = layer->width;
			int input_width = layer->input_width;
			//Output vector
			for (int k = 0; k < layer_width; k++) { //print output
				//std::cout << outputs[i][k] << ", ";
			}
			//std::cout << "\n";
			//Neuron weights
			for (int j = 0; j < layer_width; j++) { //for-each neuron
				//std::cout << "*N-" << j << " w[]  ";
				for (int k = 0; k < input_width; k++) { //print weights
					//std::cout << layer->neurons[j]->weights[k] << ", ";
				}
				//std::cout << "\n";
			}
		}
		return std::move(outputs);
	}
	void train(std::vector<float> test, std::vector<float> label) {
		//PREP
		const float learning_rate = 0.25;
		std::vector<std::vector<float>> output = compute(test);
		//std::cout << "train-network\n";
		for (int layer_index = layers.size()-1; layer_index >= 0;
				layer_index--) {
			//std::cout << "layer_index\t|" << layer_index << "\n";
			//prepare layer input
			std::vector<float> input_values = {};
			if (layer_index == 0) //INPUT layer
				input_values = test; //Grab the test-set input
			else for (size_t input_k = 0; input_k < layers
					.at(layer_index-1)->neurons.size()-1;
					input_k++) {
				auto& tmp = layers.at(layer_index-1)
					->neurons.at(input_k);
				input_values.emplace_back(tmp->output);
			}
			//std::cout << " INPUT\t";
			int tmp = 0;
			for (auto& inval : input_values) {
				//std::cout << inval << ", ";
			}
			//prepare upper layer
			std::shared_ptr<layer> upper_layer = NULL;
			if (layer_index+1 != (int)layers.size())
				upper_layer = layers.at(layer_index+1);
			//
			layers.at(layer_index)->train(input_values, label, 
				learning_rate, upper_layer);
		}
	}
	int width,depth;
	std::vector<std::shared_ptr<layer>> layers; //0 = first-layer, last is output; input 'layer' is just the vector given
};


int main(void) {
	//preparations
	std::srand(time(NULL));
	const int width = 2;
	const int depth = 3;
	network n(2, width,depth); //the network
	//show weights before
	for (int lid = n.layers.size()-1; lid >= 0; lid--) {
		std::cout << "\nLayer " << lid << ":\n========\nWeights\t|";
		for (int a = 0; a < n.layers.at(lid)->input_width; a++)
			std::cout << a << "\t";
		for (int nid = n.layers.at(lid)->neurons.size()-1;
				nid >= 0; nid--) {
			std::cout << "\nNode " << nid << "\t|";
			for (auto& w : n.layers.at(lid)->neurons.at(nid)->weights)
				std::cout << w << "\t";
		}
		std::cout << "\n";
	}

	std::vector<std::vector<float>> tests = {
		{0,0},{0,1},
		{1,0},{1,1}

	};
	std::vector<std::vector<float>> expectations {
		{0},{1},{1},{0}
	};
	for (size_t cycle = 0; cycle < 256; cycle++) {
		for (size_t idx = 0; idx < tests.size(); idx++) {
			n.train(tests[idx], expectations[idx]);
		}
	}
	std::cout << "\n\nafter";
	for (int lid = n.layers.size()-1; lid >= 0; lid--) {
		std::cout << "\nLayer " << lid << ":\n========\nWeights\t|";
		for (int a = 0; a < n.layers.at(lid)->input_width; a++)
			std::cout << a << "\t";
		for (int nid = n.layers.at(lid)->neurons.size()-1;
				nid >= 0; nid--) {
			std::cout << "\nNode " << nid << "\t|";
			for (auto& w : n.layers.at(lid)->neurons.at(nid)->weights)
				std::cout << w << "\t";
		}
		std::cout << "\n";
	}
	return 0;
}
