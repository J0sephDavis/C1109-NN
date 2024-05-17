#include "headers.hh"
class perceptron {
	public:
		perceptron(int,bool);
		virtual float calculate(const std::vector<float>);
		virtual float activation(float);
		virtual void revealWeights();
		virtual void train(float, std::vector<float>);
	protected:
		std::vector<float> weights;
		float weighted_sum = 0;
		float error_contribution = 0;
		float output = 0.0; //the last output of this node.
		float derivative = 0;
};
class bias_perceptron : public perceptron {
public:
	bias_perceptron();
	float calculate(const std::vector<float>) override;
	void revealWeights() override;
	void train(float, std::vector<float>) override;
};
class pass_perceptron : public perceptron {
public:
	pass_perceptron();
	float calculate(const std::vector<float>) override;
	void train(float, std::vector<float>) override;
};
enum perceptron_type {
	logistic,
	bias,
	passthrough
};
