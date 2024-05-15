//#define PRINT_TRAINING
#include <cstdio>
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
				weights.push_back((std::rand()% 20 + 1)*0.05);
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
		void revealWeights() {
			for (auto& w : weights) {
				std::cout << w << ",";
			}
			std::cout << "\n";
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
enum perceptron_type {
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
	layer(int width, int input_width, perceptron_type type = logistic) {
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
	/*layer::train()
	 * Computes error contribution/strength for each neuron in the layer,
	 * due to the layer they feed into.
	 * Calls the perceptrons training function. */
	virtual void train(std::vector<float> input, std::vector<float> label,
			const float learning_rate,
			std::shared_ptr<layer> upper_layer) {
		(void)label; //unused variable warning

		for (size_t u_j = 0; u_j < neurons.size(); u_j++) {
			//d_pj = f_j'(z_pj) * SUM_k (d_pk * w_kj)
			auto& neuron = neurons.at(u_j);
		//==Caclulate 2nd half of error equation - SUM_k (d_pk * w_kj==
			float associated_error = 0;
			for (size_t k = 0; k < upper_layer->neurons.size(); k++) {
				auto& uppper_neuron = upper_layer->neurons.at(k);
				//d_pk * w_kj
				associated_error += uppper_neuron
					->error_contribution * uppper_neuron
					->weights.at(u_j);
			}
			neuron->error_contribution = associated_error
				* neuron->derivative; //d_pk
#ifdef PRINT_TRAINING
			std::cout << "N-" << u_j << " d: "
				<< neuron->error_contribution << ", w: "; 
			neuron->revealWeights();
#endif

			neuron->train(learning_rate, (input));
		}
	}
	std::vector<std::unique_ptr<perceptron>> neurons;
	int width; //neurons in layer
	int input_width; //inputs to layer. 0 ifeq input layer.
};
class output_layer : public layer {
public:
	output_layer(int width, int input_width, perceptron_type type = logistic)
		: layer(width,input_width,type){
		//
	}
	void train(std::vector<float> input, std::vector<float> label,
			const float learning_rate,
			std::shared_ptr<layer> upper_layer) override {
		(void)upper_layer;
		//get error contribution of output nodes
		int label_index = 0;
		for (auto& neuron : neurons) {
			float error_strength
				=label.at(label_index++) - neuron->output;
			neuron->error_contribution = error_strength * neuron->derivative;

#ifdef PRINT_TRAINING
			std::cout << "N-" << label_index-1 << " d: "
				<< neuron->error_contribution << ", w: "; 
			neuron->revealWeights();
#endif
			//d_pj = (t_pj - o_pj) * f'(z_pj)
			//t_pj - target output
			//o_pj - actual output (in this case of the output node)
			//z_pj - weighted sum of inputs
			//derivative of logistic = f(x)(1-f(x))
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
		std::vector<std::vector<float>> outputs;
		outputs.push_back(layers[0]->output(input));
		for (int i = 1; i < depth; i++) {
			outputs.push_back(layers[i]->output(outputs[i-1]));
		}
		return std::move(outputs);
	}
	void revealWeights() {
		for (int lid = layers.size()-1; lid >= 0; lid--) {
			std::cout << "\nLayer " << lid << ":\n========\nWeights\t|";
			for (int a = 0; a < layers.at(lid)->input_width; a++)
				std::cout << a << "\t";
			for (int nid = layers.at(lid)->neurons.size()-1;
					nid >= 0; nid--) {
				std::cout << "\nNode " << nid << "\t|";
				for (auto& w : layers.at(lid)->neurons.at(nid)->weights)
					std::cout << w << "\t";
			}
			std::cout << "\n";
		}
	}
	void train(float learning_rate, std::vector<float> test, std::vector<float> label) {
		//PREP
		std::vector<std::vector<float>> output = compute(test);
		//Backpropagate
		for (int layer_index = layers.size()-1; layer_index >= 0;
				layer_index--) {
#ifdef PRINT_TRAINING
			std::cout << "# Layer-" << layer_index << "\n";
#endif
			//prepare layer input
			std::vector<float> input_values = {};
			if (layer_index == 0) 		//INPUT layer
				input_values = test;
			else for (size_t input_k = 0; input_k < layers
					.at(layer_index-1)->neurons.size()-1;
					input_k++) {
				auto& tmp = layers.at(layer_index-1)
					->neurons.at(input_k);
				input_values.emplace_back(tmp->output);
			}
			//prepare upper layer
			std::shared_ptr<layer> upper_layer = NULL;
			if (layer_index+1 != (int)layers.size())
				upper_layer = layers.at(layer_index+1);
			//TRAIN
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
	n.revealWeights();
	std::vector<std::vector<float>> tests = {
		{0,0},{0,1},
		{1,0},{1,1}

	};
	std::vector<std::vector<float>> expectations {
		{0},{1},{1},{0}
	};
	for (size_t i = 0; i < tests.size(); i++){
		std::cout << "T:[";
		for (auto& v : tests[i]) std::cout << v << ", ";
		std::cout << "], ";
		float actual = n.compute(tests[i]).back()[0];
		std::cout << "E:" << expectations.at(i)[0] << ", "
			<< "A:" << actual << ",("
			<< (expectations[i][0]-actual)<< ")\n";

	}
	std::cout << "train?";
	getchar();
#define EPOCH 1000
#define MILLENIUM 100
	for (size_t cent = 0; cent < MILLENIUM; cent++) {
		for (size_t year = 0; year < EPOCH; year++) {
			for (size_t idx = 0; idx < tests.size(); idx++) {
				n.train(0.5,tests[idx], expectations[idx]);
			}
		}
//		std::cout << "c:" << cent;
//		n.revealWeights();
//		getchar();
	}
	std::cout << "\n\nafter";
	n.revealWeights();
	for (size_t i = 0; i < tests.size(); i++){
		std::cout << "T:[";
		for (auto& v : tests[i]) std::cout << v << ", ";
		std::cout << "], ";
		float actual = n.compute(tests[i]).back()[0];
		std::cout << "E:" << expectations.at(i)[0] << ", "
			<< "A:" << actual << ",("
			<< (expectations[i][0]-actual)<< ")\n";

	}
	return 0;
}
