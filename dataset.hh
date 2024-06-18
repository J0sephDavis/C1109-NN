#ifndef DATASET_HEADER
#define DATASET_HEADER
#include <cstddef> //size_t
#include <vector>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <iostream>

class training_instance {
public:
	training_instance(size_t instance_len, size_t label_len, std::vector<float> row);
	std::vector<float> data;
	std::vector<float> label;
};

class data_file {
public:
	data_file(size_t instance_len, size_t label_len, std::filesystem::path filePath);
	std::vector<std::shared_ptr<training_instance>> data;
	size_t instance_len, label_len;
};
#endif
