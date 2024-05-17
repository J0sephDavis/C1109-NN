//#define PRINT_TRAINING
#define PRINT_ERRCON
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
		perceptron(int count_inputs, bool rand_weights = true) {
			if (rand_weights) for (int i = 0; i < count_inputs; i++) {
				weights.push_back((std::rand()% 20 -9)*0.05);
			}
			else for (int i = 0; i < count_inputs; i++) {
				weights.push_back(1);
			}
		}
		virtual float calculate(const std::vector<float> input) {
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
		virtual void revealWeights() {
			for (auto& w : weights) {
				std::cout << w << ",";
			}
		}
		//given an array of inputs, determines the weight changes needed
		virtual void train(float learning_rate, std::vector<float> input) {
			//DELTA RULE - the change in weight (from neuron u_i to u_j) is
			//  equal to learning rate, multiplied by
			//  error contribution (d_pj) then multiplied by the output of
			//  neuron-i (u_i)
			for (size_t weight_index = 0;
					weight_index < weights.size();
					weight_index++) {
				float delta_weight = learning_rate *
					error_contribution *
					input[weight_index];
				weights.at(weight_index) += delta_weight;
#ifdef PRINT_TRAINING
				std::cout << "dw-" << weight_index
					<< ":" << delta_weight << ", ";
				std::cout << "\n";
#endif
			}
		}
		std::vector<float> weights;
		float weighted_sum = 0;
		float error_contribution = 0;
		float output = 0.0; //the last output of this node.
		float derivative = 0;
};
class bias_perceptron : public perceptron {
public:
	bias_perceptron() : perceptron(0) {
		output = 1;
		derivative = 0;
	}
	float calculate(const std::vector<float> input) override {
		(void)input;
		return output;
	}
	void revealWeights() override {
		std::cout << "bias=" << output << "\n";
	}
	void train(float learning_rate, std::vector<float> input) override {
		(void) input;
		//TODO revisit
		output = output - learning_rate*error_contribution;
		return;
	}
};
class pass_perceptron : public perceptron {
public:
	pass_perceptron() : perceptron(1, true) {
		derivative = 1; //TODO prove
	}
	float calculate(const std::vector<float> input) override {
		return input[0]; //passthrough. This will take special consideration when passing input
	}
	void train(float learning_rate, std::vector<float> input) override {
		//DELTA RULE - the change in weight (from neuron u_i to u_j) is
		//  equal to learning rate, multiplied by
		//  error contribution (d_pj) then multiplied by the output of
		//  neuron-i (u_i)
		float delta_weight = learning_rate * error_contribution *
			input[0];
		weights.at(0) += delta_weight;
#ifdef PINING
		std::cout << "dw-" << weight_index
			<< ":" << delta_weight << ", ";
		std::cout << "\n";
#endif
	}
};
enum perceptron_type {
	logistic,
	bias,
	passthrough
};
//a layer of perceptrons
class layer {
public:
	/*
	 * width - the number of neurons in this layer
	 * input_width - number of neurons in previous layer.
	 * 	The accepted number of inputs for this layer.
	 */
	layer(size_t _width, size_t _input_width, size_t _bias_neurons,
			perceptron_type type = logistic) {
		this->width = _width+_bias_neurons;
		this->input_width = _input_width;
		this->bias_neurons = _bias_neurons;
		size_t neuron_index = 0;
		for (; neuron_index < bias_neurons; neuron_index++) {
			neurons.emplace_back(new bias_perceptron());
		}
		if (type == logistic) for (; neuron_index < width;
				neuron_index++) {
			neurons.emplace_back(new perceptron(input_width));
		}
		else if (type == passthrough) for (; neuron_index < width;
				neuron_index++) {
			neurons.emplace_back(new pass_perceptron());
		}
		else throw std::runtime_error("INVALID ACTIVATION TYPE");
	};
	std::vector<float> output(std::vector<float> input) {
		std::vector<float> out = {};
		size_t neuron_index = 0;
		for (; neuron_index < bias_neurons; neuron_index++)
			out.push_back(neurons[neuron_index]->calculate({}));
		for (; neuron_index < neurons.size(); neuron_index++) {
			out.push_back(neurons[neuron_index]->calculate(input));
		}
		return std::move(out);
	}
	/* update_err_contrib
	 * Computes error contribution/strength for each neuron in the layer,
	 * */
	virtual void update_err_contrib(std::vector<float> label,
			std::shared_ptr<layer> upper_layer) {
		(void)label;
		//for-each neuron in the current layer
		for (size_t u_j = 0; u_j < neurons.size(); u_j++) {
			//d_pj = f_j'(z_pj) * SUM_k (d_pk * w_kj)
			auto& neuron = neurons.at(u_j);
		//==Caclulate 2nd half of error equation - SUM_k (d_pk * w_kj==
			float associated_error = 0;
#ifdef PRINT_ERRCON
			std::cout << "### Neuron " << u_j
				<< "|k=" << upper_layer->bias_neurons << "\n";
			std::cout << "sum_k(d_kjw_kj)\t[";
#endif
			for (size_t k = upper_layer->bias_neurons;
					//Assumes the first neurons are always bias neurons.
					k < upper_layer->neurons.size(); k++) {
				auto& uppper_neuron = upper_layer->neurons.at(k);
				//d_pk * w_kj
				auto error = uppper_neuron
					->error_contribution * uppper_neuron
					->weights.at(u_j);
				associated_error += error;
#ifdef PRINT_ERRCON
				std::cout << associated_error << ", ";
#endif
			}
			neuron->error_contribution = associated_error
				* neuron->derivative; //d_pk
#ifdef PRINT_ERRCON
			std::cout << "]= " << associated_error << "\n";
			std::cout << "derivative\t|" << neuron->derivative << "\n";
			std::cout << "E contribution\t|"
				<< neuron->error_contribution << "\n";
			std::cout << "w\t[";
			neuron->revealWeights();
			std::cout << "]\n";
#endif
		}
	}
	/*layer::train()
	 * Calls the perceptrons training function of each neuron. */
	virtual void train(std::vector<float> input, const float learning_rate) {
		for (auto& neuron : neurons)
			neuron->train(learning_rate, (input));
	}
	std::vector<std::unique_ptr<perceptron>> neurons;
	size_t width; //neurons in layer
	size_t input_width; //inputs to layer. 0 ifeq input layer.
	size_t bias_neurons;
};
class output_layer : public layer {
public:
	/* No bias neuron on this layer
	 * */
	output_layer(int width, int input_width, perceptron_type type = logistic)
		: layer(width,input_width,0,type){
		//
	}
	void update_err_contrib(std::vector<float> label,
				std::shared_ptr<layer> upper_layer) override {
		(void) upper_layer;
		for (size_t j = 0; j < neurons.size(); j++) {
			const auto& neuron = neurons.at(j);
			//t_pj - target output
			//o_pj - actual output (in this case of the output node)
			float err_str = label.at(j) - neuron->output;
			//d_pj = (t_pj - o_pj) * f'(z_pj)
			neuron->error_contribution = err_str * neuron->derivative;
#ifdef PRINT_ERRCON
			std::cout << "### Neuron " << j
				<< " d: " << neuron->error_contribution
				<< "\nerr_str: " << err_str
				<< ", w: "; 
			neuron->revealWeights();
			std::cout << "\n";
#endif
		}
	}
	void train(std::vector<float> input, const float learning_rate)override {
		//get error contribution of output nodes
		for (size_t j = 0; j < neurons.size(); j++) {
			const auto& neuron = neurons.at(j);
			neuron->train(learning_rate, (input));
		}
	}
};
class input_layer : public layer {
public:
	input_layer(size_t input_width, perceptron_type type = passthrough) :
		layer(input_width, input_width, 0, type) {
		//
	};
	std::vector<float> output(std::vector<float> input) {
		std::vector<float> out = {};
		for (size_t i = 0; i < neurons.size(); i++) {
			out.push_back(neurons[i]->calculate({input[i]}));
		}
		return std::move(out);
	}
};
class network {
#define BIAS_NEURONS 1
public:
	network(int input_width, int _width, int _depth) {
		this->depth = _depth;
		this->width = _width;
		//generate
		layers.emplace_back(new input_layer(input_width, passthrough));
		for (int i = 1; i < depth-1; i++) {
//layer(size_t width, size_t input_width, perceptron_type type = logistic)
			layers.emplace_back(new layer(width, layers[i-1]->width,
						BIAS_NEURONS,logistic));
		}
		layers.emplace_back(new output_layer(1, layers[depth-2]->width, logistic)); //single output node
	}
	//compute output of network given input
	std::vector<std::vector<float>> compute(std::vector<float> input) {
		std::vector<std::vector<float>> outputs;
		std::shared_ptr<input_layer> inpl = std::dynamic_pointer_cast<input_layer>(layers[0]);
		outputs.push_back(inpl->output(input));
		for (int i = 1; i < depth; i++) {
			outputs.push_back(layers[i]->output(outputs[i-1]));
		}
		return std::move(outputs);
	}
	void revealWeights() {
		for (int lid = layers.size()-1; lid >= 0; lid--) {
			std::cout << "\nLayer " << lid
				<< ": (inputs:"	<< layers[lid]->input_width
				<< ", width" << layers[lid]->width
				<< ", bias " << layers[lid]->bias_neurons << ")";
			std::cout << "\n========\nWeights\t|";
			for (size_t a = 0; a < layers.at(lid)->input_width; a++)
				std::cout << a << "\t";
			for (int nid = layers.at(lid)->neurons.size()-1;
					nid >= 0; nid--) {
				std::cout << "\nNode " << nid << "\t|";
				layers.at(lid)->neurons.at(nid)->revealWeights();
			}
			std::cout << "\n";
		}
	}
	void train(float learning_rate, std::vector<float> test, std::vector<float> label) {
		(void)label;
		//PREP
		std::vector<std::vector<float>> output = compute(test);
#ifdef PRINT_TRAINING
		std::cout << "\n# BEGIN\n";
		std::cout << "T:\t[";
		for (auto& i : test) std::cout << i << ", ";
		std::cout << "]\n";
		std::cout << "L:\t[";
		for (auto& i : label) std::cout << i << ", ";
		std::cout << "]\n";
		std::cout << "A:\t[";
		for(auto& i : output.back()) std::cout << i << ", ";
		std::cout << "]\n";
		std::cout << "FULL OUTPUT [\n";
		for (int layer_out = output.size()-1; layer_out >= 0; layer_out--) {
			std::cout << "\t[";
			for (auto& e : output.at(layer_out)) {
				std::cout << e << ", ";
			}
			std::cout << "]\n";

		}
		std::cout << "]\n";
#endif
		std::vector<float> input_values = {};
		//Backpropagate
		for (int layer_index = layers.size()-1; layer_index >= 0;
				layer_index--) {
			//prepare layer input
			if (layer_index == 0) 		//INPUT layer
				input_values = test;
			else {
				input_values = output.at(layer_index-1);
			}
			//prepare upper layer
			std::shared_ptr<layer> upper_layer = NULL;
			if (layer_index+1 != (int)layers.size())
				upper_layer = layers.at(layer_index+1);
#ifdef PRINT_TRAINING
			std::cout << "\n## Layer " << layer_index << "\n";
			std::cout << "I:\t[";
			for (auto& i : input_values) std::cout << i << ", ";
			std::cout << "]\n";
			if (upper_layer != NULL)
				std::cout << "out_width:\t|" << upper_layer->width;
			std::cout << "\n";
#endif
			//TRAIN
			layers.at(layer_index)->update_err_contrib(label, upper_layer);
		}
		for (int layer_index = layers.size()-1; layer_index >= 0;
				layer_index--) {
			if (layer_index == 0) input_values = test;
			else input_values = output.at(layer_index-1);
			layers.at(layer_index)->train(input_values, learning_rate);
		}
	}
	int width,depth;
	std::vector<std::shared_ptr<layer>> layers; //0 = first-layer, last is output; input 'layer' is just the vector given
};


int main(void) {
	//preparations
	std::srand(time(NULL));
	const int width = 2;
	const int depth = 4;
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
#define CYCLES 1
	for (size_t cycle = 0; cycle < CYCLES; cycle++) {
		for (size_t idx = 0; idx < tests.size(); idx++) {
			n.train(0.9,tests[idx], expectations[idx]);
		}
	}
	std::cout << "trained\n";
	for (size_t i = 0; i < tests.size(); i++){
		std::cout << "T:[";
		for (auto& v : tests[i]) std::cout << v << ", ";
		std::cout << "], ";
		float actual = n.compute(tests[i]).back()[0];
		std::cout << "E:" << expectations.at(i)[0] << ", "
			<< "A:" << actual << ",("
			<< (expectations[i][0]-actual)<< ")\n";

	}
	getchar();
	n.revealWeights();
	return 0;
}
