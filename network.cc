#include "network.hh"
network::network(const neurons::hyperparams params, neurons::type neuron_t,
	const data_file& df,
	size_t _width, size_t _depth)
: params(std::move(params)) {
	//TODO warn when width < instance_len
	this->depth = _depth;
	this->width = _width;
	//generate
	layers.emplace_back(std::make_shared<input_layer>(df.instance_len, BIAS_NEURONS));
	for (size_t i = 1; i < depth - 1; i++) {
		layers.emplace_back(std::make_shared<layer>(width, layers[i-1]->width, BIAS_NEURONS,
			neuron_t,neurons::weightParams(neurons::he,neurons::normal)));
	}
	//output layer
	layers.emplace_back(
		std::make_shared<output_layer>(df.label_len, layers[depth-2]->width, neurons::logistic,
			neurons::weightParams(neurons::le_cun, neurons::normal)));
	//get some testing data
	std::random_device rand_device; //non-deterministic
	std::mt19937 random_engine(rand_device()); //PRNG
	//TODO evaluate using rand() instead of a distribution
	std::uniform_int_distribution<size_t> dist(0,df.data.size()-1); 

	//index of testing data
	const auto TEST_SIZE = df.data.size() * TESTING_RATIO;
	std::vector<size_t> testing_index(TEST_SIZE);
	//generate distribution of testing data
	for (size_t i = 0; i < TEST_SIZE ; i++)
		testing_index.push_back(dist(random_engine));
	//copy all matching indexes to training data, otherwise copy to testing data
	std::cout << "testing data{\n";
	for (size_t i = 0; i < df.data.size(); i++) {
		if (std::ranges::find(testing_index.begin(), testing_index.end(),i) != testing_index.end()) {
			testingData.push_back(df.data.at(i));
			//printout
			for (const auto& v : df.data.at(i)->data)
				std::cout << v << "\t";
			for (const auto& v : df.data.at(i)->label)
				std::cout << v << "\t";
			std::cout << "\n";
		}
		else trainingData.push_back(df.data.at(i));
	}
	std::cout << "\n}\n";
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
	std::random_device rand_device; 
	std::mt19937 random_engine(rand_device());
	std::uniform_int_distribution<size_t> dist(0,trainingData.size()-1);
const float training_ratio = 0.6f;	
	for (size_t i = 0; i < trainingData.size()*training_ratio; i++) {
//		train_on_instance(i);
		train_on_instance(dist(rand_device));
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
//RMSE(X,h) = sqrt( (1/m) * SUM_i=1:m(h(x_i)-y_i)^2
//sqrt m^-1 * S_m(h(x)- y)62
//this implementation is Mean Absolute Error for each element in the vector,
//followed by an average of all of the errors
	const size_t output_len = testingData.at(0)->label.size();
	std::vector<std::vector<float>> classifier_results(output_len);
	for (const auto& test : testingData) {
		std::vector<float> actual = std::move(compute(test->data).back());
		//TODO warn if actual.size() != test.label.size()
		float average = 0.0f;
		for (size_t idx = 0 ; idx < actual.size(); idx++) {
			classifier_results.at(idx).push_back(
					std::abs(actual.at(idx)- test->label.at(idx)));
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
