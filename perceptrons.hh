#ifndef PERCEPTRON_HEADER_HH
#define PERCEPTRON_HEADER_HH
#include "definitions.hh"
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
namespace neurons {
//the activation function or neuron type
enum type {
	logistic,
	bias,
	passthrough,
	hyperbolic_tangent,
	selection_pass,
	ReLU
};
//The initialization method for generating a weight distribution
enum weight_initializer {
	//do not use the weight initializer. weights will be{1,...} or {}
	skip_weights,
	//random_default: std::rand()
	random_default, 
	//Glorot/Xavier: NORMAL distribution, Variance=1/fan_avg, mean=0
	//or UNIFORM distribution from [-r,r], r=sqrt(3/fan_avg)
	//Good for tanh,logistic,softmax
	glorot,
	//LeCun: Glorot/Xavier but fan_avg replaced by fan_in. Variance=1/fan_in
	//GOod for SELU
	le_cun, 
	//He: Variance=2/_fan_in
	//Good for ReLU & variants
	he,
};
//the distribution to use for the weight initializer
enum distribution_type {
	uniform,
	normal
};
//the parameters for generating a weight distribution
typedef struct weightParams {
	weightParams(weight_initializer winit_t, distribution_type dist_t) {
		this->winit_t = winit_t;
		this->dist_t = dist_t;
	}
	weight_initializer winit_t;
	distribution_type dist_t;
} weightParams;
//creates a weight distribution using the initializer method & distribution type
std::vector<float> weight_distribution(const weightParams& params,
		size_t fan_in, size_t fan_out);

//The hyperparmeters that define the training behavior
typedef struct hyperparams {
	hyperparams(const float learning_rate, const float momentum) {
		this->learning_rate = learning_rate;
		this->momentum = momentum;
	}
	float momentum = 0.0f;
	float learning_rate = 0.0f;
} hyperparams;

//the weight change formula to be used during training
static inline float calc_dw(const hyperparams& params, float err_contrib,
		float input, float previous_dw) {
	//DELTA RULE - the change in weight (from neuron u_i to u_j) is
	//  equal to learning rate, multiplied by
	//  error contribution (d_pj) then multiplied by the output of
	//  neuron-i (u_i)
	float delta_weight = (params.learning_rate * err_contrib * input)
		+ (params.momentum * previous_dw);
#ifdef PRINT_TRAINING_DATA
	std::cout << "\t" << delta_weight;
	if (delta_weight == 0)
		std::cout << "(" << err_contrib << "," << input << ")";
#endif
	return delta_weight;
}
//
class perceptron {
	public:
		//TODO include output_lenght for fan_avg calc
		perceptron(const weightParams& w_params, size_t input_length,
			size_t output_len);
		type _type = logistic;
	//forward pass
		//compute sum_i (w_ij * o_ij) - called during a forward pass
		float calculate(const std::vector<float> input);
		//the activation function
		virtual float activation(float net_input);
		std::vector<float> weights;
	//training
		/*train() is virtual for the bias function which changes
		 * its output value, not input weights*/
		//called during a backward pass
		virtual void train(const hyperparams& params, std::vector<float> input);
		//the error contribution calculated during backdiff
		float error_contribution = 0.0f;
		//the output of the node during the forward pass
		float output = 0.0f;
		//the derivative of the output during the forward pass: do/dz
		float derivative = 0.0f;
		//changes in weights during the previous training session
		std::vector<float> delta_weights;
};

class perceptron_ReLU : public perceptron {
public:
	perceptron_ReLU(const weightParams& w_params,size_t input_length,
			size_t output_len);
	virtual float activation(float net_input) override;
};

class perceptron_htan : public perceptron {
public:
	perceptron_htan(const weightParams& w_params,size_t input_length,
			size_t output_len);
	virtual float activation(float net_input) override;
};

class perceptron_bias : public perceptron {
public:
	perceptron_bias();
	float activation(float net_input) override;
	void train(const hyperparams& params, std::vector<float> input) override;
private:
	float delta_output = 0.0f;
};

class perceptron_pass : public perceptron {
public:
	perceptron_pass(size_t net_input_width);
	//activation function f(z) = z
	float activation(float net_input) override;
};

// prevents training of unselected weights
class perceptron_select : public perceptron_pass {
public:
	perceptron_select(size_t net_input_width, std::vector<bool> selection_vector);
	std::vector<bool> selection_vector;
	//only train the selected inputs weights
	void train(const hyperparams& params, std::vector<float> input) override;
};

//returns a shared ptr to the constructed neuron
std::shared_ptr<perceptron> neuron_factory(
		const type neuron_t,
		const weight_initializer weight_t,
		const distribution_type dist_t,
		const size_t argument0, //used in selector
		const size_t fan_in, const size_t fan_out=0);
//returns a shared ptr to the constructed neuron
std::shared_ptr<perceptron> neuron_factory(
		const type neuron_t,
		const weightParams weight_p,
		const size_t argument0, //used in selector
		const size_t fan_in, const size_t fan_out=0);
}
#endif
