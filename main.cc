#include "csv_handler.hh"
#include "headers.hh"
#include "layers.hh"
#include <string>

const std::vector<std::vector<float>> tests {
	{0,0},{0,1},
	{1,0},{1,1}

};
const std::vector<std::vector<float>> expectations {
	{0},{1},{1},{0}
};

class network {
public:
	network(int input_width, int _width, int _depth, perceptron_type neuron_t) {
		this->depth = _depth;
		this->width = _width;
		//generate
		layers.emplace_back(new input_layer(input_width, BIAS_NEURONS));
		for (int i = 1; i < depth - 1; i++) {
			layers.emplace_back(new layer(width, layers[i-1]->width,
						BIAS_NEURONS,neuron_t));
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

typedef struct era_description {
	float sum = 0.0f;
	float average = 0.0f;
	float max = 0.0f;
	std::vector<csv_cell> cells;
	const std::string fields = "e1,e2,e3,e4,avg";
	era_description(std::vector<float> current)
	{
		for (const auto& val : current) {
			if (max < val) max = val;
			sum += val;
		}
		average = sum / 4;
		for (const float& e : current) cells.emplace_back(csv_cell(e)); 
		cells.emplace_back(average);
	}
} era_description;

int main(void) {
	//preparations
	const int width = 2;
	const int depth = 3;
	size_t run_id = 0; // for naming files
	auto srand_seed = SEED_VAL; // std::time(NULL)
	static const std::vector<float> LR {0.25,0.50,0.75};
	static const std::vector<float> MOMENTUM {0.25,0.5};
	static const std::vector<perceptron_type> types
	{logistic, hyperbolic_tanget};

for (auto& neuron_type : types)
for (auto& learning_rate : LR)
for (auto& momentum : MOMENTUM) {
	std::string fileName = "/mnt/tmpfs/csv/out" + std::to_string(run_id++) + ".csv";
	std::vector<std::string> headers = {
		sheet_description(0,0,0,(perceptron_type)0).fields,
		era_description({0.0,0.0,0.0,0.0}).fields
	};
	csv_file DATA(std::move(fileName), std::move(headers));

	std::srand(srand_seed);
	srand_seed = std::time(NULL);
	network n(2, width,depth, neuron_type); //the network
	std::vector<std::vector<float>> results = {};
	sheet_description parameterDATA(learning_rate, momentum, THRESHOLD,
			neuron_type);
	//
	for (size_t era = 0; era < MAX_ERAS; era++) {
		//1. Compute output
		results.emplace_back(n.benchmark()); 
		//2. Analyze
		era_description eraDATA(results.back());
		//3. Store
		std::vector<csv_cell> cells;
		cells.insert(cells.end(), parameterDATA.cells.begin(), parameterDATA.cells.end());
		cells.insert(cells.end(), eraDATA.cells.begin(), eraDATA.cells.end());
		DATA.add_row(std::move(cells));
		//4. Train
		//Good learners end early
		if (eraDATA.average < THRESHOLD and eraDATA.max < THRESHOLD)
			break;
		for (size_t epoch = 0; epoch < EPOCHS; epoch++) {
			for (size_t idx = 0; idx < tests.size(); idx++) {
				n.train(learning_rate, momentum,
						tests[idx], expectations[idx]);
		}}
	}
}
	return 0;
}
