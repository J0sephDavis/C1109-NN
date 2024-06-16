#include "headers.hh"
enum perceptron_type {
	UNDEFINED,
	logistic,
	bias,
	passthrough,
	hyperbolic_tangent,
	selection_pass
};
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
		perceptron(int,bool rand_weights = true);
		perceptron_type type = UNDEFINED;
		//returns the activation functions name. (for logging)
		const std::string type_str() const;
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
	float delta_output = 0.0f; // change in output during training(n-1)
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
