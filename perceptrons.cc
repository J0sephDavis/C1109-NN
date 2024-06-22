#include "perceptrons.hh"
namespace neurons {
//INITIALIZE
perceptron::perceptron(int count_inputs, bool rand_weights) {
	if (rand_weights) for (int i = 0; i < count_inputs; i++) {
		std::random_device rand_device;
		std::mt19937 random_engine(rand_device());
		//le cun initialization
		float r = 1.0f/count_inputs;
		std::normal_distribution<> dist(0,r);
		weights.push_back(
			dist(rand_device)
		);
		delta_weights.push_back(0);
	}
	else for (int i = 0; i < count_inputs; i++) {
		weights.push_back(1);
		delta_weights.push_back(0);
	}
	_type = logistic;
}
perceptron_htan::perceptron_htan(int count_inputs, bool rand_weights)
	: perceptron(std::move(count_inputs), std::move(rand_weights)) {
		_type = hyperbolic_tangent;
	}
bias_perceptron::bias_perceptron() : perceptron(0) {
		output = 1;
		derivative = 1;
		_type = bias;
}
pass_perceptron::pass_perceptron(size_t net_input_width)
	: perceptron(net_input_width, false) {
	derivative = 1; //set again during activation()
	_type = passthrough;
}
select_perceptron::select_perceptron(size_t net_input_width, std::vector<bool> selection_vector):
	pass_perceptron(net_input_width),
	selection_vector(std::move(selection_vector)) {
		_type = selection_pass;
		//This doesn't need to be done. Added for when viewing the weight dump
		for (size_t i = 0; i < selection_vector.size(); i++) {
			weights.at(i) = selection_vector.at(i);
		}
	}
//CALCULATE
float perceptron::calculate(const std::vector<float> input) {
	float weighted_sum = 0;
	if (_type==bias) return activation(weighted_sum);
	for (size_t i = 0; i < input.size(); i++) {
		weighted_sum += input[i] * weights[i];
	}
	return activation(weighted_sum);
};
//ACTIVATE
float perceptron::activation(float input) {
	//logistic sigmoid: f(x) = 1/1+e^-x
	output = 1/(1+exp(-input)); //possible underflow
	//f'(x) = f(x)(1-f(x);
	derivative = output * (1 - output);
	return output;
}
float perceptron_htan::activation(float input) {
	//dtanh(x)/dx = sech^2(x) = 1/cosh^2(x)
	output = tanh(input);//(2/(1+exp(-2*input)))-1;
	derivative = cosh(input);
	derivative = 1/(derivative * derivative);
	return output;
}
float bias_perceptron::activation(float input) {
	(void)input; //remove pedantic warning
	derivative = 1;
	return output;
}
float pass_perceptron::activation(float input) {
	output = input;
	derivative = 1; //do/dz = 1
	return output;
}
//TRAIN
//given an array of inputs, determines the weight changes needed
void perceptron::train(const hyperparams& params, std::vector<float> input) {
	for (size_t weight_index = 0; weight_index < weights.size(); weight_index++) {
		float delta_weight = calc_dw(params, error_contribution,
			input[weight_index], delta_weights[weight_index]);
		weights.at(weight_index) += delta_weight;
		delta_weights.at(weight_index) = delta_weight;
	}
}
void select_perceptron::train(const hyperparams& params, std::vector<float> input) {
	for (size_t weight_index = 0; weight_index < weights.size(); weight_index++) {
		if (selection_vector.at(weight_index) == false) continue; //skip training
		float delta_weight = calc_dw(params, error_contribution,
				input[weight_index], delta_weights[weight_index]);
		weights.at(weight_index) += delta_weight;
		delta_weights.at(weight_index) = delta_weight;
	}
}
void bias_perceptron::train(const hyperparams& params, std::vector<float> input) {
	(void) input; //removes pedantic warning of unused variable
	//change in output is the 
	delta_output = calc_dw(params, error_contribution, 1.0f, delta_output);
	output += delta_output;
}
}
