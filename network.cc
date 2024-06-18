#include "network.hh"

class network {
public:
	network(const hyperparams params, perceptron_type neuron_t,
			size_t input_width, size_t _width, size_t _depth):
	params(std::move(params)) {
		this->depth = _depth;
		this->width = _width;
		//generate
		layers.emplace_back(new input_layer(input_width, BIAS_NEURONS));
		for (int i = 1; i < depth - 1; i++) {
			layers.emplace_back(new layer(width, layers[i-1]->width, BIAS_NEURONS,neuron_t));
		}
		layers.emplace_back(new output_layer(1, layers[depth-2]->width, neuron_t)); //single output node
	}
	//compute output of network given input
	std::vector<std::vector<float>> compute(std::vector<float> input) {
		std::vector<std::vector<float>> outputs;
		for (int i = 0; i < depth; i++) {
			if (i != 0) input = outputs[i-1];
			outputs.push_back(layers[i]->output(input));
		}
		return std::move(outputs);
	}
	void train(std::vector<float> test, std::vector<float> label) {
		//PREP
		std::vector<std::vector<float>> output = compute(test);
		std::vector<float> input_values = {};
		//Backpropagate
		for (int layer_index = layers.size()-1; layer_index >= 0;
				layer_index--) {
			//prepare layer input
			if (layer_index == 0) 		//INPUT layer
				input_values = test;
			else {
				input_values = output.at(layer_index-1);
			}
			//prepare upper layer
			std::shared_ptr<layer> upper_layer = NULL;
			if (layer_index+1 != (int)layers.size())
				upper_layer = layers.at(layer_index+1);
			layers.at(layer_index)->update_err_contrib(label, upper_layer);
		}
		for (int layer_index = layers.size()-1; layer_index >= 0;
				layer_index--) {
			if (layer_index == 0) input_values = test;
			else input_values = output.at(layer_index-1);
			layers.at(layer_index)->train(input_values, params);
		}
	}
	std::vector<float> benchmark() {
		std::vector<float> results = {};
		for (size_t i = 0; i < tests.size(); i++){
			float actual = compute(tests[i]).back()[0];
			results.push_back(abs(expectations[i][0]-actual));
		}
		return std::move(results);
	}
	std::string weight_header() const {
		std::stringstream out;
		for (size_t lid = 0; lid < layers.size(); lid++) {
			const auto& neurons = layers.at(lid)->neurons;
			for (size_t nid = 0; nid < neurons.size();nid++) {
				const auto& neuron = neurons.at(nid);
				if (neuron->type == bias) {
					out << "L" << lid << "N" << nid << "B" << neuron->output << ",";
				}
				else {
					const auto& weights = neuron->weights;
					for (size_t wid = 0; wid < weights.size(); wid++) {
						out << "L" << lid << "N" << nid << "W" << wid << ",";
					}
				}
			}
		}
		return out.str();
	}
	std::vector<csv_cell> weights() {
		std::vector<csv_cell> rv = {};
		for (const auto& layer : layers) {
			for (const auto& neuron : layer->neurons) {
				switch(neuron->type) {
					case(bias):
						rv.push_back(neuron->output);
						break;
					default:
						rv.insert(rv.end(), neuron->weights.begin(), neuron->weights.end());
				}
			}
		}
		return std::move(rv);
	}
	size_t width,depth;
	std::vector<std::shared_ptr<layer>> layers;
	const hyperparams params; //hyper parameters that define the training
};

typedef struct sheet_description {
	const std::string fields = "learning rate,momentum,threshold,type";
	std::vector<csv_cell> cells;
	sheet_description(float learning_rate, float momentum, float threshold,
			perceptron_type type) {
		cells.emplace_back(learning_rate);
		cells.emplace_back(momentum);
		cells.emplace_back(threshold);
		cells.emplace_back((int)type);
	}
} sheet_description;
