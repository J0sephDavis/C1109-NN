#include "layers.hh"
//INITIALIZE
layer::layer(size_t _width, size_t _input_width, size_t _bias_neurons,
		neurons::type type, neurons::weightParams weight_p) {
	//width is the number of neurons of the passed type to create. bias neurons are tacked onto the width during initialization, not when calling the constructor
	//input width is the width of the previous layer
	this->width = _width+_bias_neurons;
	this->input_width = _input_width;
	this->bias_neurons = _bias_neurons;
	size_t neuron_index = 0;
	//first, initialize all bias neurons.
	for (; neuron_index < bias_neurons; neuron_index++) {
		neurons.emplace_back(neurons::neuron_factory(neurons::bias,
			neurons::weightParams(neurons::skip_weights,neurons::normal),0,0));
	}
	//Then, initialize neurons of the layer type
	for (; neuron_index < width; neuron_index++) {
		//for the selection vector
		int arg0 = neuron_index - bias_neurons;
		neurons.emplace_back(neurons::neuron_factory(type,weight_p,arg0,this->input_width));
	}
};
output_layer::output_layer(int width, int input_width, neurons::type type, neurons::weightParams weight_p)
	: layer(width,input_width,0,type,weight_p){
	//NO BIAS NEURONS
}
input_layer::input_layer(size_t input_width, size_t bias_neurons)
	: layer(input_width, input_width, bias_neurons, neurons::selection_pass,
			neurons::weightParams(neurons::skip_weights, neurons::normal)) {
	//
}
//OUTPUT
std::vector<float> layer::output(std::vector<float> input) {
	std::vector<float> out = {};
	size_t neuron_index = 0;
	for (; neuron_index < neurons.size(); neuron_index++) {
		out.push_back(neurons[neuron_index]->calculate(input));
	}
	return std::move(out);
}
std::vector<float> input_layer::output(std::vector<float> input) {
	std::vector<float> out = {};
	size_t neuron_index = 0;
	for (; neuron_index < neurons.size(); neuron_index++) {
		out.push_back(neurons[neuron_index]->calculate(input));
	}
	return std::move(out);
}
//ERROR_CONTRIBUTION
float layer::get_associated_err(size_t neuron_j) {
	float associated_error = 0;
	//For all neurons, this upper layer, that are not bias neurons
#ifdef PRINT_ERR_CONTRIBUTION
	std::cout << "get_associated(" << neuron_j << "):";
#endif
	for (size_t k = bias_neurons; k < neurons.size(); k++) {
		const auto& u_k = neurons.at(k);
#ifdef PRINT_ERR_CONTRIBUTION
		std::cout << "(" << u_k->error_contribution << "*"
			<< u_k->weights[neuron_j] << ") + ";
#endif
		associated_error +=
			u_k->error_contribution * u_k->weights.at(neuron_j);
	}
#ifdef PRINT_ERR_CONTRIBUTION
	std::cout << "= " << associated_error << "\n";
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
#ifdef PRINT_ERR_CONTRIBUTION
		if (neuron->error_contribution == 0) {
			auto associated = upper_layer->get_associated_err(u_j);
			if (associated == 0) continue;
			auto deriv = neuron->derivative;
			std::cout << "[" << u_j << "](" << associated << "," << deriv<< ")\n";
		}
		else
			std::cout << "[" << u_j << "]: " << neuron->error_contribution << "\n";
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
#ifdef PRINT_ERR_CONTRIBUTION
		std::cout << "output_err_" << j << ": " << err_str << "\n";
#endif
		//d_pj = (t_pj - o_pj) * f'(z_pj)
		neuron->error_contribution = err_str * neuron->derivative;
	}
}
//TRAIN
/* Calls the perceptrons training function of each neuron. */
void layer::train(std::vector<float> input, const neurons::hyperparams& params) {
	auto x = 0;
	for (auto& neuron : neurons) {
#ifdef PRINT_TRAINING_DATA
		std::cout << "N_" << x++;
		neuron->train(params, (input));
		std::cout << "\n";
#else
		neuron->train(params, (input));
#endif
	}
}
void output_layer::train(std::vector<float> input, const neurons::hyperparams& params) {
	//get error contribution of output nodes
	auto x = 0;
	for (auto& neuron : neurons) {
#ifdef PRINT_TRAINING_DATA
		std::cout << "O_" << x++;
		neuron->train(params, input);
		std::cout << "\n";
#else
		neuron->train(params, input);
#endif
	}
}
