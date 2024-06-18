#include "dataset.hh"
training_instance::training_instance(size_t instance_len, size_t label_len, std::vector<float> row)
	: data(instance_len), label(label_len){
	std::copy_n(row.begin(), instance_len+1, data.begin());
	std::copy_n(row.begin()+instance_len, label_len, label.begin());
}
data_file::data_file(size_t instance_len, size_t label_len, std::filesystem::path filePath) {
	std::ifstream file(filePath);
	if (file.bad()) throw std::runtime_error("Failed to open // data_file.bad()");
	std::cout << "File: " << filePath.string() << "\n";
	const size_t row_len = instance_len + label_len;
	while (true) {
		std::vector<float> row(row_len);
		std::string buff;
		for (size_t idx = 0; i < row_len-1; i++) {
			std::getline(file,buff,','); //get n-1 comma separated values
			//eof->incomplete/non-existant record. do not save this one
			if (file.eof()) break;
			//TODO should it be std::stof(std::move(buff)) or std::stof(buff)?
			row.emplace_back(std::stof(std::move(buff)));
		}
		if (file.eof()) break;
	//get the final value which is unlikely to have a comma before the new line.
	//TODO check if buff ends with a separator and delete it before std::stof(buff)
		std::getline(file,buff); 
		row.emplace_back(std::stof(std::move(buff)));
		//add training instance to vector
		data.emplace_back(training_instance(instance_len, label_len, std::move(row)));

	}
}
