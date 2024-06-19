#include "network.hh"
#include "dataset.hh"
network::network(const hyperparams params, perceptron_type neuron_t,
	const data_file& df,
	size_t _width, size_t _depth)
: params(std::move(params)) {
	//TODO warn when width < instance_len
	this->depth = _depth;
	this->width = _width;
	//generate
	layers.emplace_back(new input_layer(df.instance_len, BIAS_NEURONS));
	for (size_t i = 1; i < depth - 1; i++) {
		layers.emplace_back(
			new layer(width, layers[i-1]->width, BIAS_NEURONS,neuron_t));
	}
	 //single output node
	layers.emplace_back(new output_layer(df.label_len, layers[depth-2]->width, neuron_t));
	//get some testing data
	std::cout << "testing data{\n";
	for (size_t i = 0; i < df.data.size() * 0.2; i++) {
		auto random_index = std::rand() % df.data.size();
		testingData.push_back(df.data.at(random_index));
		for (const auto& v : df.data.at(random_index)->data) {
				std::cout << v << "\t";
		}
		for (const auto& v : df.data.at(random_index)->label) {
				std::cout << v << "\t";
		}
		std::cout << "\n";
	}
	std::cout << "\n}\n";
	//train on whole dataset for now
	for (const auto& sp_inst : df.data) {
		trainingData.push_back(sp_inst);
	}
}

std::vector<std::vector<float>> network::compute(std::vector<float> input) {
	std::vector<std::vector<float>> outputs;
	for (size_t i = 0; i < depth; i++) {
		if (i != 0) input = outputs[i-1];
		outputs.push_back(layers[i]->output(input));
	}
	return std::move(outputs);
}

void network::train() {
	for (size_t i = 0; i < trainingData.size(); i++) {
		train_on_instance(i);
	}
}

void network::train_on_instance(size_t instance_id) {
//PREP
	//compute the forward pass of the network
	std::vector<std::vector<float>> output_data = compute(trainingData.at(instance_id)->data);
	//add the input to the beginning of the output_data
	output_data.insert(output_data.begin(), trainingData.at(instance_id)->data);
	//Backpropagate
	//1. err contribution
	for (size_t layer_index = layers.size()-1; layer_index != 0; layer_index--) {
		std::shared_ptr<layer> upper_layer = NULL;
		if (layer_index+1 < layers.size()) {
			upper_layer = layers.at(layer_index+1);

		}
		//update the error contribution
		layers.at(layer_index)->update_err_contrib(
				std::cref(trainingData.at(instance_id)->label), upper_layer);
	}
	//2. gradient descent
	for (size_t layer_index = layers.size()-1; layer_index !=0; layer_index--) {
		layers.at(layer_index)->train(
				std::cref(output_data.at(layer_index)),
				std::cref(params));
	}
}
/*TODO for classification, return a vector with an output len equal to
 * the length of the test label. Each index will refer to each classification
 * so that the model can be properly graded*/
std::vector<float> network::benchmark() {
	const size_t output_len = testingData.at(0)->label.size();
	std::vector<std::vector<float>> classifier_results(output_len);
	for (const auto& test : testingData) {
		std::vector<float> actual = std::move(compute(test->data).back());
		//TODO warn if actual.size() != test.label.size()
		float average = 0.0f;
		for (size_t idx = 0 ; idx < actual.size(); idx++) {
			classifier_results.at(idx).push_back(
					std::abs(test->label.at(idx) - actual.at(idx)));
		}
	}
	std::vector<float> results;
	for (const auto& cResults : classifier_results) {
		float average = 0.0f;
		for (size_t i = 0; i < cResults.size(); i++) {
			average += cResults.at(i);
		}
		average /= cResults.size();
		results.push_back(std::move(average));
	}
	return std::move(results);
}
std::string network::weight_header() const {
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
std::vector<csv_cell> network::weights() {
	std::vector<csv_cell> rv = {};
	for (const auto& layer : layers) {
		for (const auto& neuron : layer->neurons) {
			switch(neuron->type) {
				case(bias):
					rv.push_back(neuron->output);
					break;
				default:
					rv.insert(rv.end(),
						neuron->weights.begin(),
						neuron->weights.end());
			}
		}
	}
	return std::move(rv);
}
