#include "perceptrons.hh"
namespace neurons {
std::shared_ptr<perceptron>neuron_factory(const type neuron_t,
		const weight_initializer weight_t, const distribution_type dist_t,
		const size_t arugment0, const size_t fan_in, const size_t fan_out) {
	return neuron_factory(neuron_t,
			std::move(weightParams(std::move(weight_t),std::move(dist_t))),
			arugment0, fan_in, fan_out);
}
std::shared_ptr<perceptron>neuron_factory(const type neuron_t, const weightParams weight_p,
		const size_t argument0,	const size_t fan_in, const size_t fan_out) {
	std::shared_ptr<perceptron> neuron;
	switch (neuron_t) {
		case(logistic):
			neuron = std::make_shared<perceptron>
				(perceptron(std::move(weight_p), fan_in, fan_out));
			break;
		case(bias):
			neuron = std::make_shared<perceptron>(perceptron_bias());
			break;
		case(passthrough):
			neuron = std::make_shared<perceptron>
				(perceptron_pass(fan_in));
			break;
		case(hyperbolic_tangent):
			neuron = std::make_shared<perceptron>
				(perceptron_htan(std::move(weight_p), fan_in, fan_out));
			break;
		case(selection_pass):
			std::vector<bool> selector(fan_in);
			for (size_t i = 0; i < fan_in; i++) {
				if (i == argument0) selector.push_back(true);
				else selector.push_back(false);
			}
			neuron = std::make_shared<perceptron>
				(perceptron_select(fan_in, std::move(selector)));
			break;
	}
	return std::move(neuron);
}
//INITIALIZE
perceptron::perceptron(const weightParams& w_params, size_t input_len,
		size_t output_len = 0) {
	//create the delta weight vector (for momentum)
	for (size_t i = 0; i < input_len; i++)
		delta_weights.push_back(0);
	
	//early return for nodes who handle weights in their own constructors
	if (w_params.winit_t == skip_weights) {
		for (size_t i = 0; i < input_len; i++) {
			weights.push_back(1);
		}
	}
	//create a distribution of weights
	else {
		this->weights = std::move(
			//TODO breakpoint to test the pass by ref to ptr
			weight_distribution(w_params,
				input_len,output_len)
		);
	}
}

perceptron_htan::perceptron_htan(const weightParams& w_params,
		size_t input_len, size_t output_len = 0)
	: perceptron(std::move(w_params), std::move(input_len), std::move(output_len)) {
		_type = hyperbolic_tangent;
	}
perceptron_bias::perceptron_bias() : perceptron(std::move(weightParams(skip_weights,normal)), 0) {
		output = 1;
		derivative = 1;
		_type = bias;
}
perceptron_pass::perceptron_pass(size_t net_input_width)
	: perceptron(std::move(weightParams(skip_weights, normal)),net_input_width) {
	derivative = 1; //set again during activation()
	_type = passthrough;
}
perceptron_select::perceptron_select(size_t net_input_width, std::vector<bool> selection_vector):
	perceptron_pass(net_input_width),
	selection_vector(std::move(selection_vector)) {
		_type = selection_pass;
		//This doesn't need to be done. Added for when viewing the weight dump later
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
float perceptron_bias::activation(float input) {
	(void)input; //remove pedantic warning
	derivative = 1;
	return output;
}
float perceptron_pass::activation(float input) {
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
void perceptron_select::train(const hyperparams& params, std::vector<float> input) {
	for (size_t weight_index = 0; weight_index < weights.size(); weight_index++) {
		if (selection_vector.at(weight_index) == false) continue; //skip training
		float delta_weight = calc_dw(params, error_contribution,
				input[weight_index], delta_weights[weight_index]);
		weights.at(weight_index) += delta_weight;
		delta_weights.at(weight_index) = delta_weight;
	}
}
void perceptron_bias::train(const hyperparams& params, std::vector<float> input) {
	(void) input; //removes pedantic warning of unused variable
	//TODO could get rid of delta_output & use the delta_weights vector...
	delta_output = calc_dw(params, error_contribution, 1.0f, delta_output);
	output += delta_output;
}
std::vector<float> weight_distribution(const weightParams& params,
		//mixing references and pointers will bite me the ass one of these days
		size_t fan_in, size_t fan_out) {
	std::vector<float> weights;
	std::random_device rand_device;
	std::mt19937 random_engine(rand_device());

	float variance, mean, r;
	size_t fan_avg = (fan_in+fan_out)/2;

	switch(params.winit_t) {
		case(skip_weights):
			//this code should never be reached because skip_weights is handled
			//in percetron::perceptron(...)
			return {};
		case(random_default):
			//TODO understand how random_engine works. should be PRNG?
			for (size_t i = 0; i < fan_in; i++)
				weights.push_back(random_engine());
			break;
		case(glorot):
			variance = 1.0f/(fan_avg);
			mean = 0.0f;
			r = sqrt(3.0f/fan_avg);
			break;
		case(le_cun):
			variance = 1.0f/(fan_in);
			mean = 0.0f;
			r = sqrt(3.0f/fan_in);
			break;
		case(he):
			//TODO what does r=?
			variance = 2.0f/(fan_in);
			mean = 0.0f;
			r = sqrt(3.0f/fan_in);
			break;
	}
	switch(params.dist_t) {
		case(uniform):
			{
			std::uniform_real_distribution<> dist(-r,r);
			for (size_t i = 0; i < fan_in; i++)
				weights.push_back(dist(rand_device));
			}
			break;
		case(normal):
			{
			std::normal_distribution<> dist(0,r);
			for (size_t i = 0; i < fan_in; i++)
				weights.push_back(dist(rand_device));
			}
			break;
	}
	return std::move(weights);
}
}
