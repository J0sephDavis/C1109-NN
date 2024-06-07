#include "csv_handler.hh"
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
	network(int input_width, int _width, int _depth, perceptron_type neuron_t) {
		this->depth = _depth;
		this->width = _width;
		//generate
		layers.emplace_back(new input_layer(width, input_width, BIAS_NEURONS));
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
	float learning_rate, momentum, threshold;
	perceptron_type neuron_type;
	sheet_description(float l, float m, float t, perceptron_type n) {
		learning_rate=l;
		momentum=m;
		threshold=t;
		neuron_type=n;
	}
	void print() {
		std::cout << learning_rate << "," << momentum << ","
			<< threshold << ",";
		if (neuron_type == logistic)
			std::cout << "logistic";
		else if (neuron_type == hyperbolic_tanget)
			std::cout << "tanh";
		else
			std::cout << neuron_type;
	}
} sheet_description;

typedef struct era_description {
	float sum;
	float average;
	float max;
	std::vector<float> error;
	era_description(std::vector<float> current)
		: error(current)
	{
		for (size_t idx = 0; idx < current.size(); idx++) {
			const float& val = current.at(idx);
			if (max < val) max = val;
			average += val*0.25;
			sum += val;
		}
		average = sum / current.size();
	}
	void print() {
		for (auto& v : error) std::cout << v << ",";
		std::cout << max
			<< "," << sum
			<< "," << average;
	}
	static const std::string fields = "e1,e2,e3,e4,avg";
	std::vector<csv_cell> getCells() {
		std::vector<csv_cell> row = {};
		for (const auto& e : error) row.emplace_back(csv_cell(e)); 
		row.emplace_back(average);
		return std::move(row);
	}	
} era_description;

int main(void) {
	//preparations
	const int width = 2;
	const int depth = 3;
	size_t run_id = 0;
	auto srand_seed = SEED_VAL; // std::time(NULL)
	static const std::vector<float> LR {0.25,0.5,0.75};
	static const std::vector<float> MOMENTUM {0.25,0.50,0.75};
	static const std::vector<perceptron_type> types
	{logistic, hyperbolic_tanget};
for (auto& neuron_type : types)
for (auto& learning_rate : LR)
for (auto& momentum : MOMENTUM) {
	std::srand(srand_seed);
	srand_seed = std::time(NULL);
	network n(2, width,depth, neuron_type); //the network
	std::vector<std::vector<float>> results = {};
	sheet_description current_run(learning_rate, momentum, THRESHOLD,
			neuron_type);
	//
	for (size_t era = 0; era < MAX_ERAS; era++) {
		//1. Compute output
		//2. Analyze
		//3. Store
		//4. Train
		results.emplace_back(n.benchmark()); 
		bool last = false;
		era_description current_era(results.back());
		//catch great learners
		if (current_era.average < THRESHOLD and current_era.max < THRESHOLD)
			era = MAX_ERAS;
		if (era+1 >= MAX_ERAS)
			last = true;
		//
		if (last) continue; //not much different than break in this case.
		for (size_t epoch = 0; epoch < EPOCHS; epoch++) {
			for (size_t idx = 0; idx < tests.size(); idx++) {
				n.train(learning_rate, momentum,
						tests[idx], expectations[idx]);
		}}
	}
}
	return 0;
}
