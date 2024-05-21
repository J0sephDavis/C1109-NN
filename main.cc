#include "headers.hh"
#include "layers.hh"

const std::vector<std::vector<float>> tests {
	{0,0},{0,1},
	{1,0},{1,1}

};
const std::vector<std::vector<float>> expectations {
	{0},{1},{1},{0}
};

class network {
public:
	network(int input_width, int _width, int _depth) {
		this->depth = _depth;
		this->width = _width;
		//generate
		layers.emplace_back(new input_layer(width, input_width, BIAS_NEURONS));
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
		for (size_t lid = 0; lid < layers.size(); lid++) {
			for (size_t nid = 0;nid < layers.at(lid)->neurons.size();
					nid++) {
				layers.at(lid)->neurons.at(nid)->revealWeights();
				std::cout << "\n";
			}
		}
	}
	void train(const float learning_rate, const float momentum, std::vector<float> test, std::vector<float> label) {
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
			layers.at(layer_index)->update_err_contrib(label, upper_layer);
		}
		for (int layer_index = layers.size()-1; layer_index >= 0;
				layer_index--) {
			if (layer_index == 0) input_values = test;
			else input_values = output.at(layer_index-1);
			layers.at(layer_index)->train(input_values, learning_rate, momentum);
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
	int width,depth;
	std::vector<std::shared_ptr<layer>> layers;
};

typedef struct sheet_description {
	float learning_rate, momentum, threshold;
	sheet_description(float l, float m, float t) {
		learning_rate=l;
		momentum=m;
		threshold=t;
	}
	void print() {
		std::cout << learning_rate << "," << momentum << ","
			<< threshold;
	}
} sheet_description;

float print_stat(std::vector<float> input) {
	float sum = 0;
	for (auto& i : input) {
		std::cout << i << ",";
		sum+=i;
	}
	float avg = sum / input.size();
	std::cout << sum << "," << avg;
	return avg;
}
int main(void) {
	//preparations
	const int width = 2;
	const int depth = 3;
	static const std::vector<float> LR {0.1,0.25,0.50,0.75,1.0};
	static const std::vector<float> MOMENTUM {0,0.1,0.25,0.5,0.75,1.0};
	size_t run_id = 0;
for (auto& learning_rate : LR) for (auto& momentum : MOMENTUM) {
	std::srand(time(NULL));
//	std::srand(SEED_VAL);
	network n(2, width,depth); //the network
	std::vector<std::vector<float>> results = {};
	sheet_description current_run(learning_rate, momentum, THRESHOLD);
	//
	bool last = false;
	
	results.emplace_back(n.benchmark());
	current_run.print(); std::cout << ",";
	print_stat(results.back());
	std::cout << "," << last << "\n";

	size_t total_epochs = 0;
	for (size_t era = 0; era < MAX_ERAS; era++) {
		for (size_t epoch = 0; epoch < EPOCHS; epoch++, total_epochs++) {
			for (size_t idx = 0; idx < tests.size(); idx++) {
				n.train(learning_rate, momentum,
						tests[idx], expectations[idx]);
			}
		}
		results.emplace_back(n.benchmark()); 
		current_run.print(); std::cout << ",";
		float average  = print_stat(results.back());
		if (average < THRESHOLD) {
			era = MAX_ERAS;
			last = true;
		}
		else if (era+1 >= MAX_ERAS) {
			last = true;
		}
		std::cout << "," << last << "\n";
	}
//	std::cout << "END\n";
}
	return 0;
}
