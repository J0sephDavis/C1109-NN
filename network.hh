#ifndef NN_HEADER
#define NN_HEADER
#include <random>
#include "csv_handler.hh"
#include "dataset.hh"
#include "layers.hh"

class network {
public:
	network(const neurons::hyperparams params, neurons::type neuron_t,
			const data_file& dataFile,
			size_t _width, size_t _depth);
	//compute output of network given input
	std::vector<std::vector<float>> compute(std::vector<float> input);
	//train on dataset
	void train();
	std::vector<float> benchmark();
	size_t width,depth;
	std::vector<std::shared_ptr<layer>> layers;
	const neurons::hyperparams params; 
	//training data
	std::vector<std::shared_ptr<training_instance>> trainingData;
	//testing data
	std::vector<std::shared_ptr<training_instance>> testingData;
private:
	//train on specific instance
	void train_on_instance(size_t instance_id);
};
#endif
