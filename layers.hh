#include "perceptrons.hh"
//a layer of perceptrons
class layer {
public:
	layer(size_t,size_t,size_t,perceptron_type);
	virtual std::vector<float> output(std::vector<float> input);
	float get_associated_err(size_t neuron_j);
	virtual void update_err_contrib(std::vector<float> label,
			std::shared_ptr<layer> upper_layer);
	virtual void train(std::vector<float>input, const float learning_rate);
public:
	std::vector<std::unique_ptr<perceptron>> neurons;
	size_t width; //neurons in layer
	size_t input_width; //inputs to layer. 0 ifeq input layer.
	size_t bias_neurons;
};
class output_layer : public layer {
public:
	/* No bias neuron on this layer */
	output_layer(int width, int input_width,
			perceptron_type type = logistic);
	void update_err_contrib(std::vector<float> label,
				std::shared_ptr<layer> upper_layer) override;
	void train(std::vector<float> input,
			const float learning_rate) override;
};
class input_layer : public layer {
public:
	input_layer(size_t input_width, perceptron_type type = passthrough);
	std::vector<float> output(std::vector<float> input) override;
};
