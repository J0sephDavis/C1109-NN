#include "perceptrons.hh"
//INITIALIZE
perceptron::perceptron(int count_inputs, bool rand_weights) {
	if (rand_weights) for (int i = 0; i < count_inputs; i++) {
		weights.push_back((std::rand()% 20 -9)*0.05);
	}
	else for (int i = 0; i < count_inputs; i++) {
		weights.push_back(1);
	}
}
bias_perceptron::bias_perceptron() : perceptron(0) {
		output = 1;
		derivative = 0;
}
pass_perceptron::pass_perceptron() : perceptron(1, true) {
	derivative = 1;
}
//CALCULATE
float perceptron::calculate(const std::vector<float> input) {
	for (size_t i = 0; i < input.size(); i++) {
		weighted_sum += input[i] * weights[i];
	}
	return activation(weighted_sum);
};
float bias_perceptron::calculate(const std::vector<float> input) {
	(void)input;
	return output;
}
float pass_perceptron::calculate(const std::vector<float> input) {
	return input[0]; //passthrough. This will take special consideration when passing input
}
//ACTIVATE
float perceptron::activation(float input) {
	//logistic sigmoid

	//f(x) = 1/1+e^-x
	output = 1/(1+exp(-input)); //possible underflow
	//f'(x) = f(x)(1-f(x);
	derivative = output * (1 - output);
	return output;
}
//REAVEAL
void perceptron::revealWeights() {
	for (auto& w : weights) {
		std::cout << w << ",";
	}
}
void bias_perceptron::revealWeights() {
	std::cout << "bias=" << output << "\n";
}
//TRAIN
//given an array of inputs, determines the weight changes needed
void perceptron::train(float learning_rate, std::vector<float> input) {
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
void pass_perceptron::train(float learning_rate, std::vector<float> input) {
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
void bias_perceptron::train(float learning_rate, std::vector<float> input) {
	(void) input;
	//TODO revisit
	output = output - learning_rate*error_contribution;
	return;
}
