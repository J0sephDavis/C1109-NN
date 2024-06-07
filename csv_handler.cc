#include "csv_handler.hh"
#include <iostream>
#include <sstream>
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
std::string csv_cell::get() const {
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
csv_file::csv_file(std::string file_path, std::vector<std::string> headers)
	: file(file_path, std::ios_base::out) {
	if (file.bad() or not file.is_open())
		throw std::runtime_error("failed to open csv");
	if (!fields.empty()) {
		fields = std::move(headers);
		for (const auto& field : fields) {
			file << field << ",";
		}
		file << "\n";
	}
}
void csv_file::add_row(std::vector<csv_cell> cells) {
	if (!fields.empty() and cells.size() > fields.size())
		std::cerr << "cells > fields";
	//
	std::ostringstream row_stream;
	for (const auto& cell : cells) {
		row_stream << cell.get();
	}
	row_stream << "\n";
	file << row_stream.str();
}
