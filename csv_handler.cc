#include "csv_handler.hh"
//==csv_cell==
csv_cell::csv_cell(int v) { set(v); }
csv_cell::csv_cell(float v) { set(v); };
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
	case(FLOAT):
		return std::to_string(value.f);
	default:
		return "ERR";
}
}
//==csv_file==
csv_file::csv_file(std::string file_path, std::vector<std::string> headers)
	: file(file_path, std::ios_base::out) {
	if (file.bad() or not file.is_open())
		throw std::runtime_error("failed to open csv");
	for (const auto& field : headers) {
		file << field << ",";
	}
	file << "\n";
	
}
void csv_file::add_row(std::vector<csv_cell> cells) {
	std::ostringstream row_stream;
	for (const auto& cell : cells) {
		row_stream << cell.get() << ",";
	}
	row_stream << "\n";
	file << row_stream.str();
}
