//a single perceptron - logistic
class perceptron {
	public:
		perceptron(int count_inputs, bool rand_weights = true) {
			if (rand_weights) for (int i = 0; i < count_inputs; i++) {
				weights.push_back((std::rand()% 20 -9)*0.05);
			}
			else for (int i = 0; i < count_inputs; i++) {
				weights.push_back(1);
			}
		}
		virtual float calculate(const std::vector<float> input) {
			for (size_t i = 0; i < input.size(); i++) {
				weighted_sum += input[i] * weights[i];
			}
			return activation(weighted_sum);
		};
		virtual float activation(float input) {
			//logistic sigmoid

			//f(x) = 1/1+e^-x
			output = 1/(1+exp(-input)); //possible underflow
			//f'(x) = f(x)(1-f(x);
			derivative = output * (1 - output);
			return output;
		}
		virtual void revealWeights() {
			for (auto& w : weights) {
				std::cout << w << ",";
			}
		}
		//given an array of inputs, determines the weight changes needed
		virtual void train(float learning_rate, std::vector<float> input) {
			//DELTA RULE - the change in weight (from neuron u_i to u_j) is
			//  equal to learning rate, multiplied by
			//  error contribution (d_pj) then multiplied by the output of
			//  neuron-i (u_i)
			for (size_t weight_index = 0;
					weight_index < weights.size();
					weight_index++) {
				float delta_weight = learning_rate *
					error_contribution *
					input[weight_index];
				weights.at(weight_index) += delta_weight;
#ifdef PRINT_TRAINING
				std::cout << "dw-" << weight_index
					<< ":" << delta_weight << ", ";
				std::cout << "\n";
#endif
			}
		}
		std::vector<float> weights;
		float weighted_sum = 0;
		float error_contribution = 0;
		float output = 0.0; //the last output of this node.
		float derivative = 0;
};
class bias_perceptron : public perceptron {
public:
	bias_perceptron() : perceptron(0) {
		output = 1;
		derivative = 0;
	}
	float calculate(const std::vector<float> input) override {
		(void)input;
		return output;
	}
	void revealWeights() override {
		std::cout << "bias=" << output << "\n";
	}
	void train(float learning_rate, std::vector<float> input) override {
		(void) input;
		//TODO revisit
		output = output - learning_rate*error_contribution;
		return;
	}
};
class pass_perceptron : public perceptron {
public:
	pass_perceptron() : perceptron(1, true) {
		derivative = 1; //TODO prove
	}
	float calculate(const std::vector<float> input) override {
		return input[0]; //passthrough. This will take special consideration when passing input
	}
	void train(float learning_rate, std::vector<float> input) override {
		//DELTA RULE - the change in weight (from neuron u_i to u_j) is
		//  equal to learning rate, multiplied by
		//  error contribution (d_pj) then multiplied by the output of
		//  neuron-i (u_i)
		float delta_weight = learning_rate * error_contribution *
			input[0];
		weights.at(0) += delta_weight;
#ifdef PINING
		std::cout << "dw-" << weight_index
			<< ":" << delta_weight << ", ";
		std::cout << "\n";
#endif
	}
};
enum perceptron_type {
	logistic,
	bias,
	passthrough
};
