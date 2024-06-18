#ifndef NN_HEADER
#define NN_HEADER
#include "csv_handler.hh"
#include "dataset.hh"
#include "layers.hh"

#define BIAS_NEURONS 1

class network {
public:
	network(const hyperparams params, perceptron_type neuron_t,
			const data_file& dataFile,
			size_t _width, size_t _depth);
	//compute output of network given input
	std::vector<std::vector<float>> compute(std::vector<float> input);
	void train(std::vector<float> test, std::vector<float> label);
//==for tabulating later==
	std::vector<float> benchmark();
	std::string weight_header() const;
	std::vector<csv_cell> weights();
//==vars==
	size_t width,depth;
	std::vector<std::shared_ptr<layer>> layers;
	const hyperparams params; 
	//training data
	std::vector<std::shared_ptr<training_instance>> trainingData;
	//testing data
	std::vector<std::shared_ptr<training_instance>> testingData;
};
#endif
