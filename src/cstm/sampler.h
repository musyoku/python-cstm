#pragma once
#include <random>

namespace cstm {
	namespace sampler {
		extern std::mt19937 mt;
		extern std::minstd_rand minstd;
		double gamma(double a, double b);
		double beta(double a, double b);
		double bernoulli(double p);
		double uniform(double min = 0, double max = 0);
		int uniform_int(int min = 0, int max = 0);
	}
}