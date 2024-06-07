#include "csv_handler.hh"
#include <stdexcept>
//==csv_cell==
void csv_cell::set(int _val) {
	value.i = _val;
	type = INT;
}
void csv_cell::set(float _val) {
	value.f = _val;
	type = FLOAT;
}
std::string csv_cell::get() {
	switch(type) {
		case(INT):
			return std::to_string(value.i);
			break;
		case(FLOAT):
			return std::to_string(value.f);
			break;
	}
}
//==csv_file==
csv_file::csv_file(std::string file_path): file(file_path, std::ios_base::out) {
	if (file.bad() or not file.is_open())
		throw std::runtime_error("failed to open csv");
}
void csv_file::add_row(std::vector<csv_cell> cells) {
	//
}
