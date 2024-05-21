//#define PRINT_ERRCON
//#define PRINT_COMPUTE
#include "layers.hh"
#include <memory>
//INITIALIZE
layer::layer(size_t _width, size_t _input_width, size_t _bias_neurons,
		perceptron_type type) {
	this->width = _width+_bias_neurons;
	this->input_width = _input_width;
	this->bias_neurons = _bias_neurons;
	size_t neuron_index = 0;
	for (; neuron_index < bias_neurons; neuron_index++) {
		neurons.emplace_back(new bias_perceptron());
	}
	if (type == logistic) for (; neuron_index < width;
			neuron_index++) {
		neurons.emplace_back(new perceptron(input_width));
	}
	else if (type == passthrough) for (; neuron_index < width;
			neuron_index++) {
		std::vector<bool> weight_mask = {};
		for (size_t i = 0; i < width; i++)
			weight_mask.push_back(i==neuron_index);
		neurons.emplace_back(new pass_perceptron(width, std::move(weight_mask)));
		// for-each bias, create an empty input in the input. e.g., {1,1} becomes {X,1,1} where X is the only bias.
	}
	else throw std::runtime_error("INVALID ACTIVATION TYPE");
};
output_layer::output_layer(int width, int input_width, perceptron_type type)
	: layer(width,input_width,0,type){
	//
}
input_layer::input_layer(size_t width, size_t input_width, size_t bias_neurons)
	: layer(width, input_width, bias_neurons, passthrough) {
	//
}
//OUTPUT
std::vector<float> layer::output(std::vector<float> input) {
#ifdef PRINT_COMPUTE
	std::cout << "Input: ";
	for (auto i : input) std::cout << i << ", ";
	std::cout << "\n";
#endif
	std::vector<float> out = {};
	size_t neuron_index = 0;
	for (; neuron_index < neurons.size(); neuron_index++) {
		out.push_back(neurons[neuron_index]->calculate(input));
#ifdef PRINT_COMPUTE
		std::cout << "N[" << neuron_index << "].calc = "
			<< out.back() << "\n";
#endif
	}
	return std::move(out);
}
std::vector<float> input_layer::output(std::vector<float> input) {
	for (size_t i = 0; i < bias_neurons; i++) {
		input.insert(input.begin(),0); //dead space for bias neuron
	}
#ifdef PRINT_COMPUTE
	std::cout << "(IL) Input: ";
	for (auto i : input) std::cout << i << ", ";
	std::cout << "\n";
#endif
	std::vector<float> out = {};
	size_t neuron_index = 0;
	for (; neuron_index < neurons.size(); neuron_index++) {
		out.push_back(neurons[neuron_index]->calculate(input));
#ifdef PRINT_COMPUTE
		std::cout << "N[" << neuron_index << "].calc = "
			<< out.back() << "\n";
#endif
	}
	return std::move(out);
}
//ERROR_CONTRIBUTION
float layer::get_associated_err(size_t neuron_j) {
#ifdef PRINT_ERRCON
	std::cout << "### Neuron " << neuron_j
		<< "|k=" << bias_neurons << "\n";
	std::cout << "sum_k(d_kjw_kj)\t[";
#endif
	float associated_error = 0;
	//For all neurons, this upper layer, that are not bias neurons
	for (size_t k = bias_neurons; k < neurons.size(); k++) {
		const auto& u_k = neurons.at(k);
#ifdef PRINT_ERRCON
		float err = u_k->error_contribution * u_k->weights.at(neuron_j);
		std::cout << err << ", ";
		associated_error += err;
#else
		associated_error +=
			u_k->error_contribution * u_k->weights.at(neuron_j);
#endif
	}
#ifdef PRINT_ERRCON
		std::cout << "]= " << associated_error << "\n";
#endif
	return associated_error;
}
/* update_err_contrib
 * Computes error contribution/strength for each neuron in the layer,
 * */
void layer::update_err_contrib(std::vector<float> label,
		std::shared_ptr<layer> upper_layer) {
	(void)label;
	//for-each neuron in the current layer
	for (size_t u_j = 0; u_j < neurons.size(); u_j++) {
		//d_pj = f_j'(z_pj) * SUM_k (d_pk * w_kj)
		auto& neuron = neurons.at(u_j);
	//==Caclulate 2nd half of error equation - SUM_k (d_pk * w_kj==
		neuron->error_contribution
			= upper_layer->get_associated_err(u_j)
			* neuron->derivative; //d_pk
#ifdef PRINT_ERRCON
		std::cout << "derivative\t|" << neuron->derivative << "\n";
		std::cout << "E contribution\t|"
			<< neuron->error_contribution << "\n";
		std::cout << "w\t[";
		neuron->revealWeights();
		std::cout << "]\n";
#endif
	}
}
void output_layer::update_err_contrib(std::vector<float> label,
			std::shared_ptr<layer> upper_layer) {
	(void) upper_layer;
	for (size_t j = 0; j < neurons.size(); j++) {
		const auto& neuron = neurons.at(j);
		//t_pj - target output
		//o_pj - actual output (in this case of the output node)
		float err_str = label.at(j) - neuron->output; //t-o
		//d_pj = (t_pj - o_pj) * f'(z_pj)
		neuron->error_contribution = err_str * neuron->derivative;
#ifdef PRINT_ERRCON
		std::cout << "### Neuron " << j
			<< " d: " << neuron->error_contribution
			<< "\nerr_str: " << err_str
			<< ", w: "; 
		neuron->revealWeights();
		std::cout << "\n";
#endif
	}
}
//TRAIN
/* Calls the perceptrons training function of each neuron. */
void layer::train(std::vector<float> input, const float learning_rate) {
	size_t neuron_index = 0;
	//for (; neuron_index < this->bias_neurons; neuron_index++) {
	//	auto n = std::dynamic_pointer_cast<bias_perceptron>(neurons.at(neuron_index));
	//	n->train(learning_rate, input);
	//}
	//for (; neuron_index < this->neurons.size(); neuron_index++) {
	//	neurons.at(neuron_index)->train(learning_rate,input);
	//}
	for (auto& neuron : neurons) {
		neuron->train(learning_rate, (input));

	}
}
void output_layer::train(std::vector<float> input,
		const float learning_rate) {
	//get error contribution of output nodes
	for (size_t j = 0; j < neurons.size(); j++) {
		const auto& neuron = neurons.at(j);
		neuron->train(learning_rate, (input));
	}
}
