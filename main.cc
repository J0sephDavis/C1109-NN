#include "definitions.hh"
#include "csv_handler.hh"
#include "network.hh"

#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <ctime>

const std::vector<std::vector<float>> tests {
	{0,0},{0,1},
	{1,0},{1,1}
};
const std::vector<std::vector<float>> expectations {
	{0},{1},{1},{0}
};
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

int main(void) {
	//preparations
	const int width = 2;
	const int depth = 3;
#ifdef SEED_VAL
	auto srand_seed = SEED_VAL;
#endif
	static const std::vector<float> LR {0.25};
	static const std::vector<float> MOMENTUM {0.1};
	static const std::vector<perceptron_type> types
	{logistic, hyperbolic_tangent};

for (auto& neuron_type : types)
for (auto& learning_rate : LR)
for (auto& momentum : MOMENTUM) {
	std::string neuronType;
	switch(neuron_type) {
		case(logistic):
			neuronType = "LOG";
			break;
		case(hyperbolic_tangent):
			neuronType = "TAN";
			break;
		default:
			neuronType = "UNK";
			break;
	}
#ifdef SEED_VAL
	std::srand(srand_seed);
#elif
	std::srand(std::time(NULL));
#endif
	hyperparams parameters(learning_rate, momentum);
	network n(parameters, neuron_type, 2, width,depth); //the network
	std::vector<std::vector<float>> results = {};
	sheet_description parameterDATA(learning_rate, momentum, THRESHOLD,
			neuron_type);
	//
	std::stringstream fileName;
	fileName << "/mnt/tmpfs/csv/"
		<< neuronType + "/"
		<< "L" << std::fixed << std::setprecision(2) << learning_rate
		<< "M" << std::fixed << std::setprecision(2) << momentum
		<< ".csv";
	std::cout << "FILENAME: " << fileName.str() << "\n";
	std::vector<std::string> headers = {
		sheet_description(0,0,0,(perceptron_type)0).fields,
		era_description({0.0,0.0,0.0,0.0}).fields,
		n.weight_header(),
	};
	csv_file DATA(std::move(fileName.str()), std::move(headers));

	for (size_t era = 0; era < MAX_ERAS; era++) {
		//1. Compute output
		results.emplace_back(n.benchmark()); 
		//2. Analyze
		era_description eraDATA(results.back());
		//3. Store
		std::vector<csv_cell> cells;
		cells.insert(cells.end(), parameterDATA.cells.begin(), parameterDATA.cells.end());
		cells.insert(cells.end(), eraDATA.cells.begin(), eraDATA.cells.end());
		const auto weights = n.weights();
		cells.insert(cells.end(), weights.begin(), weights.end());
		DATA.add_row(std::move(cells));
		//4. Train
		//Good learners end early
		if (eraDATA.average < THRESHOLD and eraDATA.max < THRESHOLD)
			break;
		for (size_t epoch = 0; epoch < EPOCHS; epoch++) {
			for (size_t idx = 0; idx < tests.size(); idx++) {
				n.train(tests[idx], expectations[idx]);
		}}
	}
}
	return 0;
}
