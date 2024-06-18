#include "dataset.hh"
#include <memory>
training_instance::training_instance(size_t instance_len, size_t label_len, std::vector<float> row)
	: data(instance_len), label(label_len){
	std::copy_n(row.begin(), instance_len+1, data.begin());
	std::copy_n(row.begin() + instance_len, label_len, label.begin());
}
data_file::data_file(size_t _instance_len, size_t _label_len, std::filesystem::path filePath) {
	this->instance_len = std::move(_instance_len);
	this->label_len = std::move(_label_len);
	std::ifstream file(filePath);
	if (file.bad()) throw std::runtime_error("Failed to open // data_file.bad()");
	std::cout << "File: " << filePath.string() << "\n";
	const size_t row_len = instance_len + label_len;
	while (true) {
		std::vector<float> row;
		std::string buff;
		for (size_t idx = 0; idx < row_len-1; idx++) {
			std::getline(file,buff,','); //get n-1 comma separated values
			//eof->incomplete/non-existant record. do not save this one
			if (file.eof()) break;
			//TODO should it be std::stof(std::move(buff)) or std::stof(buff)?
			row.emplace_back(std::stof(std::move(buff)));
		}
		//!!EXIT POINT!!
		if (file.eof()) break; 
	//get the final value which is unlikely to have a comma before the new line.
	//TODO check if buff ends with a separator and delete it before std::stof(buff)
		std::getline(file,buff); 
		row.emplace_back(std::stof(std::move(buff)));
		//shared pointer to training instance
		auto instance = std::make_shared<training_instance>
			(instance_len,label_len, std::move(row));
		//add training instance to vector
		data.push_back(std::move(instance));
	}
	std::cout << "DATA-LEN:" << data.size() << "\n";
}
