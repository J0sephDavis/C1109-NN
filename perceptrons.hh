#include "headers.hh"
enum perceptron_type {
	logistic,
	bias,
	passthrough,
	hyperbolic_tangent,
	UNDEFINED
};
//
class perceptron {
	public:
		perceptron(int,bool rand_weights = true);
		//compute sum_i (w_ij * o_ij)
		float calculate(const std::vector<float> input);
		virtual float activation(float net_input);
		virtual void train(const float momentum,
				const float learning_rate,
				std::vector<float> input);
		std::string type_str();
	public:
		std::vector<float> weights;
		float error_contribution = 0;
		float output = 0.0; //the last output of this node.
		float derivative = 0;
		perceptron_type type = UNDEFINED;
	protected:
		std::vector<float> delta_weights; //changes in weights
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
	void train(const float momentum, const float learning_rate,
			std::vector<float> input) override;
private:
	float delta_output = 0.0f; // change in output during training(n-1)
};
class pass_perceptron : public perceptron {
public:
	pass_perceptron(int net_input_width, std::vector<bool> weight_mask = {});
	float activation(float net_input) override;
	void train(const float momentum, float learning_rate,
			std::vector<float> input) override;
};
