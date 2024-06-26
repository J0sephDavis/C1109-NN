#include "network.hh"

#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iterator>

typedef struct era_description {
	float sum = 0.0f;
	float average = 0.0f;
	float max = 0.0f;
	std::vector<csv_cell> cells;
	const std::string fields = "e1,e2,e3,avg";
	era_description(std::vector<float> current)
	{
		for (const auto& val : current) {
			if (max < val) max = val;
			sum += val;
		}
		average = sum / current.size();
		for (const float& e : current) cells.emplace_back(csv_cell(e)); 
		cells.emplace_back(average);
	}
} era_description;

typedef struct sheet_description {
	const std::string fields = "learning rate,momentum,threshold,type";
	std::vector<csv_cell> cells;
	sheet_description(float learning_rate, float momentum, float threshold,
			neurons::type type) {
		cells.emplace_back(learning_rate);
		cells.emplace_back(momentum);
		cells.emplace_back(threshold);
		cells.emplace_back((int)type);
	}
} sheet_description;
namespace n=neurons;
int main(void) {
	//preparations
	data_file training_set(4,3,"/mnt/tmpfs/iris.csv");
	const int width = training_set.instance_len*5;
	const int depth = training_set.instance_len*5; //arbitrary depth chosen
#ifdef SEED_VAL
	auto srand_seed = SEED_VAL;
#endif
	static const std::vector<float> LR {0.25};
	static const std::vector<float> MOMENTUM {0.1};
	static const std::vector<n::type> types
	{n::logistic};//, n::hyperbolic_tangent};

for (auto& neuron_type : types)
for (auto& learning_rate : LR)
for (auto& momentum : MOMENTUM) {
	std::string neuronType;
	switch(neuron_type) {
		case(n::logistic):
			neuronType = "LOG";
			break;
		case(n::hyperbolic_tangent):
			neuronType = "TAN";
			break;
		case(n::ReLU):
			neuronType = "ReLU";
			break;
		default:
			neuronType = "UNK";
			break;
	}
#ifdef SEED_VAL
	std::srand(srand_seed);
#else
	std::srand(std::time(NULL));
#endif
	n::hyperparams parameters(learning_rate, momentum);
	network n(parameters, neuron_type, std::cref(training_set), width,depth); //the network
	std::vector<std::vector<float>> results = {};
	sheet_description parameterDATA(learning_rate, momentum, THRESHOLD,
			neuron_type);
	//
	std::stringstream fileName;
	fileName << "/mnt/tmpfs/csv/"
		<< neuronType + "-"
		<< "L" << std::fixed << std::setprecision(2) << learning_rate
		<< "M" << std::fixed << std::setprecision(2) << momentum
		<< ".csv";
	std::cout << "FILENAME: " << fileName.str() << "\n";
	std::vector<std::string> headers = {
		sheet_description(0,0,0,(neurons::type)0).fields,
		era_description({0.0,0.0,0.0,0.0}).fields,
	};
	csv_file DATA(std::move(fileName.str()), std::move(headers));

	for (size_t era = 0; era < MAX_ERAS; era++) {
		//1. Compute output
		results.emplace_back(n.benchmark()); 
		//2. Analyze
		era_description eraDATA(results.back());
		//3. Store
		std::vector<csv_cell> cells;
		std::copy(parameterDATA.cells.begin(), parameterDATA.cells.end(), std::back_insert_iterator(cells));
		std::copy(eraDATA.cells.begin(), eraDATA.cells.end(), std::back_insert_iterator(cells));
		DATA.add_row(std::move(cells));
		//4. Train
		//Good learners end early
		if (eraDATA.average < THRESHOLD and eraDATA.max < THRESHOLD)
			break;
		for (size_t epoch = 0; epoch < EPOCHS; epoch++) {
			n.train();
		}
		std::cout << "average:" << eraDATA.average << "\n";
//control sector
	//	char x;
	//	std::cout << " continue?";
	//	std::cin >> x;
	//	if (x == 'n') break;
	}
}
	return 0;
}
