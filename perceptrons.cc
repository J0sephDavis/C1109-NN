#include "perceptrons.hh"
//INITIALIZE
perceptron::perceptron(int count_inputs, bool rand_weights) {
	if (rand_weights) for (int i = 0; i < count_inputs; i++) {
		weights.push_back((std::rand()% 20 -10)*0.5);
		delta_weights.push_back(0);
	}
	else for (int i = 0; i < count_inputs; i++) {
		weights.push_back(1);
		delta_weights.push_back(0);
	}
}
bias_perceptron::bias_perceptron() : perceptron(0) {
		output = 1;
		derivative = 1;
}
pass_perceptron::pass_perceptron(int net_input_width, std::vector<bool> weight_mask)
	: perceptron(net_input_width, false) {
	derivative = 1;
	//if a weight mask is provided. Clear all values where the mask is 0
	if (weight_mask.empty() == false)
		for (size_t index = 0; index < weight_mask.size(); index++)
			if (weight_mask[index] == false)
				weights.at(index) = 0;
}
//CALCULATE
float perceptron::calculate(const std::vector<float> input) {
	float weighted_sum = 0;
	if (weights.empty()) return activation(weighted_sum);//bias node
	for (size_t i = 0; i < input.size(); i++) {
		weighted_sum += input[i] * weights[i];
#ifdef PRINT_COMPUTE
		std::cout << "z+= " << input[i] << " * " << weights[i] << "\n";
#endif
	}
#ifdef PRINT_COMPUTE
	std::cout << "z= " << weighted_sum << "\n";
#endif
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
float bias_perceptron::activation(float input) {
	(void)input; //disregard
	derivative = 1;
	return output;
}
float pass_perceptron::activation(float input) {
	output = input;
	derivative = 1; //do/dz = 1 ?
	return output;
}
//REAVEAL
void perceptron::revealWeights() {
	for (auto& w : weights) {
		std::cout << w << ",";
	}
}
void bias_perceptron::revealWeights() {
	std::cout << "bias=" << output;
}
//TRAIN
#define MOMENTUM 0.3f
//given an array of inputs, determines the weight changes needed
void perceptron::train(float learning_rate, std::vector<float> input) {
	//DELTA RULE - the change in weight (from neuron u_i to u_j) is
	//  equal to learning rate, multiplied by
	//  error contribution (d_pj) then multiplied by the output of
	//  neuron-i (u_i)
	for (size_t weight_index = 0;
			weight_index < weights.size();
			weight_index++) {
		float delta_weight = (learning_rate *
			error_contribution *
			input[weight_index])
			+ (MOMENTUM * delta_weights.at(weight_index));
		weights.at(weight_index) += delta_weight;
		delta_weights.at(weight_index) = delta_weight;
#ifdef PRINT_TRAINING
			std::cout << "dw-" << weight_index
				<< ":" << delta_weight << ", ";
			std::cout << "\n";
#endif
	}
}
void pass_perceptron::train(float learning_rate, std::vector<float> input) {
	for (size_t weight_index = 0; weight_index < weights.size();
			weight_index++) {
		//skip the weights intentionally left blank.
		//Could potentially be a problem if training gets
		//a weights set to 0, but unlikely?
		if (weights.at(weight_index) == 0) continue;
		float delta_weight = (learning_rate * error_contribution
			* input[weight_index])
			+ (MOMENTUM * delta_weights.at(weight_index));
		weights.at(weight_index) += delta_weight;
		delta_weights.at(weight_index) = delta_weight;
#ifdef PRINT_TRAINING
			std::cout << "dw-" << weight_index
				<< ":" << delta_weight << ", ";
			std::cout << "\n";
#endif
	}

}
void bias_perceptron::train(float learning_rate, std::vector<float> input) {
	(void) input;
	//TODO revisit
	output = output + learning_rate*error_contribution;
	return;
}
