#ifndef PERCEPTRON_HEADER_HH
#define PERCEPTRON_HEADER_HH
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <random>
namespace neurons {
enum type {
	logistic,
	bias,
	passthrough,
	hyperbolic_tangent,
	selection_pass
};
enum weight_initializer {
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
enum distribution_type {
	uniform,
	normal
};
//creates a weight distribution using the initializer method & distribution type
std::vector<float> weight_distribution(weight_initializer winit_t,
		distribution_type dist_t, size_t fan_in, size_t fan_out);

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
	//DELTA RULE - the change in weight (from neuron u_i to u_j) is
	//  equal to learning rate, multiplied by
	//  error contribution (d_pj) then multiplied by the output of
	//  neuron-i (u_i)
		float input, float previous_dw) {
	return (params.learning_rate * err_contrib * input)
		+ (params.momentum * previous_dw);
}
//
class perceptron {
	public:
		//TODO include output_lenght for fan_avg calc
		perceptron(size_t input_length,
				weight_initializer weight_t,
				distribution_type dist_t);
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

class perceptron_htan : public perceptron {
public:
	perceptron_htan(int, bool rand_weights = true);
	virtual float activation(float net_input) override;
};

class bias_perceptron : public perceptron {
public:
	bias_perceptron();
	float activation(float net_input) override;
	void train(const hyperparams& params, std::vector<float> input) override;
private:
	float delta_output = 0.0f;
};

class pass_perceptron : public perceptron {
public:
	pass_perceptron(size_t net_input_width);
	//activation function f(z) = z
	float activation(float net_input) override;
};

// prevents training of unselected weights
class select_perceptron : public pass_perceptron {
public:
	select_perceptron(size_t net_input_width, std::vector<bool> selection_vector);
	std::vector<bool> selection_vector;
	//only train the selected inputs weights
	void train(const hyperparams& params, std::vector<float> input) override;
};
}
#endif
