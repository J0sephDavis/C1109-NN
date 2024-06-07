#include <fstream>
#include <ios>
#include <ostream>
#include <vector>
#include <string>
class csv_cell {
public:
	csv_cell() {
	}
	void set(int _val);
	void set(float _val);
	//returns the value as a string
	std::string get();
private:
	enum{INT,FLOAT} type;
	union {
		int 	i;
		float 	f;
	} value;
};

class csv_file {
public:
	csv_file(std::string file_path);
	void add_row(std::vector<csv_cell>);
	//
	std::vector<std::string> fields;
	std::ofstream file;
};
