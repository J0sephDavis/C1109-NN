#include <perceptrons.hh>
//a layer of perceptrons
class layer {
public:
	/*
	 * width - the number of neurons in this layer
	 * input_width - number of neurons in previous layer.
	 * 	The accepted number of inputs for this layer.
	 */
	layer(size_t _width, size_t _input_width, size_t _bias_neurons,
			perceptron_type type = logistic) {
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
			neurons.emplace_back(new pass_perceptron());
		}
		else throw std::runtime_error("INVALID ACTIVATION TYPE");
	};
	std::vector<float> output(std::vector<float> input) {
		std::vector<float> out = {};
		size_t neuron_index = 0;
		for (; neuron_index < bias_neurons; neuron_index++)
			out.push_back(neurons[neuron_index]->calculate({}));
		for (; neuron_index < neurons.size(); neuron_index++) {
			out.push_back(neurons[neuron_index]->calculate(input));
		}
		return std::move(out);
	}
	/* update_err_contrib
	 * Computes error contribution/strength for each neuron in the layer,
	 * */
	virtual void update_err_contrib(std::vector<float> label,
			std::shared_ptr<layer> upper_layer) {
		(void)label;
		//for-each neuron in the current layer
		for (size_t u_j = 0; u_j < neurons.size(); u_j++) {
			//d_pj = f_j'(z_pj) * SUM_k (d_pk * w_kj)
			auto& neuron = neurons.at(u_j);
		//==Caclulate 2nd half of error equation - SUM_k (d_pk * w_kj==
			float associated_error = 0;
#ifdef PRINT_ERRCON
			std::cout << "### Neuron " << u_j
				<< "|k=" << upper_layer->bias_neurons << "\n";
			std::cout << "sum_k(d_kjw_kj)\t[";
#endif
			for (size_t k = upper_layer->bias_neurons;
					//Assumes the first neurons are always bias neurons.
					k < upper_layer->neurons.size(); k++) {
				auto& uppper_neuron = upper_layer->neurons.at(k);
				//d_pk * w_kj
				auto error = uppper_neuron
					->error_contribution * uppper_neuron
					->weights.at(u_j);
				associated_error += error;
#ifdef PRINT_ERRCON
				std::cout << associated_error << ", ";
#endif
			}
			neuron->error_contribution = associated_error
				* neuron->derivative; //d_pk
#ifdef PRINT_ERRCON
			std::cout << "]= " << associated_error << "\n";
			std::cout << "derivative\t|" << neuron->derivative << "\n";
			std::cout << "E contribution\t|"
				<< neuron->error_contribution << "\n";
			std::cout << "w\t[";
			neuron->revealWeights();
			std::cout << "]\n";
#endif
		}
	}
	/*layer::train()
	 * Calls the perceptrons training function of each neuron. */
	virtual void train(std::vector<float> input, const float learning_rate) {
		for (auto& neuron : neurons)
			neuron->train(learning_rate, (input));
	}
	std::vector<std::unique_ptr<perceptron>> neurons;
	size_t width; //neurons in layer
	size_t input_width; //inputs to layer. 0 ifeq input layer.
	size_t bias_neurons;
};
class output_layer : public layer {
public:
	/* No bias neuron on this layer
	 * */
	output_layer(int width, int input_width, perceptron_type type = logistic)
		: layer(width,input_width,0,type){
		//
	}
	void update_err_contrib(std::vector<float> label,
				std::shared_ptr<layer> upper_layer) override {
		(void) upper_layer;
		for (size_t j = 0; j < neurons.size(); j++) {
			const auto& neuron = neurons.at(j);
			//t_pj - target output
			//o_pj - actual output (in this case of the output node)
			float err_str = label.at(j) - neuron->output;
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
	void train(std::vector<float> input, const float learning_rate)override {
		//get error contribution of output nodes
		for (size_t j = 0; j < neurons.size(); j++) {
			const auto& neuron = neurons.at(j);
			neuron->train(learning_rate, (input));
		}
	}
};
class input_layer : public layer {
public:
	input_layer(size_t input_width, perceptron_type type = passthrough) :
		layer(input_width, input_width, 0, type) {
		//
	};
	std::vector<float> output(std::vector<float> input) {
		std::vector<float> out = {};
		for (size_t i = 0; i < neurons.size(); i++) {
			out.push_back(neurons[i]->calculate({input[i]}));
		}
		return std::move(out);
	}
};
