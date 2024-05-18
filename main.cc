#define PRINT_TRAINING
#include "headers.hh"
#include "layers.hh"
class network {
#define BIAS_NEURONS 1
public:
	network(int input_width, int _width, int _depth) {
		this->depth = _depth;
		this->width = _width;
		//generate
		layers.emplace_back(new layer(width, input_width, BIAS_NEURONS, passthrough));
		for (int i = 1; i < depth - 1; i++) {
			layers.emplace_back(new layer(width, layers[i-1]->width,
						BIAS_NEURONS,logistic));
		}
		layers.emplace_back(new output_layer(1, layers[depth-2]->width, logistic)); //single output node
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
	void revealWeights() {
		for (int lid = layers.size()-1; lid >= 0; lid--) {
			std::cout << "\nLayer " << lid
				<< ": (inputs:"	<< layers[lid]->input_width
				<< ", width" << layers[lid]->width
				<< ", bias " << layers[lid]->bias_neurons << ")";
			std::cout << "\n========\nWeights\t|";
			for (size_t a = 0; a < layers.at(lid)->input_width; a++)
				std::cout << a << "\t";
			for (int nid = layers.at(lid)->neurons.size()-1;
					nid >= 0; nid--) {
				std::cout << "\nNode " << nid << "\t|";
				layers.at(lid)->neurons.at(nid)->revealWeights();
			}
			std::cout << "\n";
		}
	}
	void train(float learning_rate, std::vector<float> test, std::vector<float> label) {
		(void)label;
		for (int i = 0; i < BIAS_NEURONS; i++) {
			test.insert(test.begin(),0.0f);
		}
		//PREP
		std::vector<std::vector<float>> output = compute(test);
#ifdef PRINT_TRAINING
		std::cout << "\n# BEGIN\n";
		std::cout << "T:\t[";
		for (auto& i : test) std::cout << i << ", ";
		std::cout << "]\n";
		std::cout << "L:\t[";
		for (auto& i : label) std::cout << i << ", ";
		std::cout << "]\n";
		std::cout << "A:\t[";
		for(auto& i : output.back()) std::cout << i << ", ";
		std::cout << "]\n";
		std::cout << "FULL OUTPUT [\n";
		for (int layer_out = output.size()-1; layer_out >= 0; layer_out--) {
			std::cout << "\t[";
			for (auto& e : output.at(layer_out)) {
				std::cout << e << ", ";
			}
			std::cout << "]\n";

		}
		std::cout << "]\n";
#endif
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
#ifdef PRINT_TRAINING
			std::cout << "\n## Layer " << layer_index << "\n";
			std::cout << "I:\t[";
			for (auto& i : input_values) std::cout << i << ", ";
			std::cout << "]\n";
			if (upper_layer != NULL)
				std::cout << "out_width:\t|" << upper_layer->width;
			std::cout << "\n";
#endif
			//TRAIN
			layers.at(layer_index)->update_err_contrib(label, upper_layer);
		}
		for (int layer_index = layers.size()-1; layer_index >= 0;
				layer_index--) {
			if (layer_index == 0) input_values = test;
			else input_values = output.at(layer_index-1);
			layers.at(layer_index)->train(input_values, learning_rate);
		}
	}
	int width,depth;
	std::vector<std::shared_ptr<layer>> layers; //0 = first-layer, last is output; input 'layer' is just the vector given
};


int main(void) {
	//preparations
	std::srand(time(NULL));
	const int width = 2;
	const int depth = 4;
	network n(2, width,depth); //the network
	//show weights before
	n.revealWeights();
	std::vector<std::vector<float>> tests = {
		{0,0},{0,1},
		{1,0},{1,1}

	};
	std::vector<std::vector<float>> expectations {
		{0},{1},{1},{0}
	};
	for (size_t i = 0; i < tests.size(); i++){
		std::cout << "T:[";
		for (auto& v : tests[i]) std::cout << v << ", ";
		std::cout << "], ";
		float actual = n.compute(tests[i]).back()[0];
		std::cout << "E:" << expectations.at(i)[0] << ", "
			<< "A:" << actual << ",("
			<< (expectations[i][0]-actual)<< ")\n";

	}
#define CYCLES 1000
#if CYCLES > 0
	for (size_t cycle = 0; cycle < CYCLES; cycle++) {
		for (size_t idx = 0; idx < tests.size(); idx++) {
			n.train(0.9,tests[idx], expectations[idx]);
		}
	}
	for (size_t i = 0; i < tests.size(); i++){
		std::cout << "T:[";
		for (auto& v : tests[i]) std::cout << v << ", ";
		std::cout << "], ";
		float actual = n.compute(tests[i]).back()[0];
		std::cout << "E:" << expectations.at(i)[0] << ", "
			<< "A:" << actual << ",("
			<< (expectations[i][0]-actual)<< ")\n";

	}
#else
	n.train(0.25, tests[0],expectations[0]);
#endif
	//getchar();
	std::cout << "trained\n";
	n.revealWeights();
	return 0;
}
