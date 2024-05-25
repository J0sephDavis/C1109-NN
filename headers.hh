#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <memory> //unique_ptr
#include <math.h>
#include <fstream>

#define MAX_ERAS 700
#define EPOCHS 1
#define THRESHOLD 0.2f
#define SEED_VAL 2809
#define BIAS_NEURONS 1
#define PRINT_CSV
#ifndef PRINT_CSV
//#define PRINT_COMPUTE
//#define PRINT_TRAINING
//#define PRINT_ERRCON
#endif
