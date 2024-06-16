#include "perceptrons.hh"
//INITIALIZE
perceptron::perceptron(int count_inputs, bool rand_weights) {
	if (rand_weights) for (int i = 0; i < count_inputs; i++) {
		weights.push_back(((std::rand()%10)-5) * 0.1);
		delta_weights.push_back(0);
	}
	else for (int i = 0; i < count_inputs; i++) {
		weights.push_back(1);
		delta_weights.push_back(0);
	}
	type = logistic;
}
perceptron_htan::perceptron_htan(int count_inputs, bool rand_weights)
	: perceptron(std::move(count_inputs), std::move(rand_weights)) {
		type = hyperbolic_tangent;
	}
bias_perceptron::bias_perceptron() : perceptron(0) {
		output = 1;
		derivative = 1;
		type = bias;
}
pass_perceptron::pass_perceptron(int net_input_width, std::vector<bool> weight_mask)
	: perceptron(net_input_width, false) {
	derivative = 1;
	//if a weight mask is provided. Clear all values where the mask is 0
	if (weight_mask.empty() == false)
		for (size_t index = 0; index < weight_mask.size(); index++)
			if (weight_mask[index] == false)
				weights.at(index) = 0;
	type = passthrough;
}
//CALCULATE
float perceptron::calculate(const std::vector<float> input) {
	float weighted_sum = 0;
	if (weights.empty()) return activation(weighted_sum);//bias node
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
void pass_perceptron::train(const hyperparams& params, std::vector<float> input) {
	for (size_t weight_index = 0; weight_index < weights.size(); weight_index++) {
		//skip the weights intentionally left blank.
		//Could potentially be a problem if training gets
		//a weights set to 0, but unlikely?
		if (weights.at(weight_index) == 0) continue;
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
//TYPE
const std::string perceptron::type_str() const {
	switch (type) {
		case (logistic):
			return "logistic";
		case(bias):
			return "bias";
		case(passthrough):
			return "passthrough";
		case(hyperbolic_tangent):
			return "htan";
		case(UNDEFINED):
		default:
			return "UNK";
	}
}
