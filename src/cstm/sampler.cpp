#include <random>
#include <chrono>
#include "sampler.h"

namespace cstm {
	namespace sampler {
		int seed = std::chrono::system_clock::now().time_since_epoch().count();
		// int seed = 1;
		std::mt19937 mt(seed);
		std::minstd_rand minstd(seed);
		double gamma(double a, double b){
			std::gamma_distribution<double> distribution(a, 1.0 / b);
			return distribution(mt);
		}
		double beta(double a, double b){
			double ga = gamma(a, 1.0);
			double gb = gamma(b, 1.0);
			return ga / (ga + gb);
		}
		double bernoulli(double p){
			std::uniform_real_distribution<double> rand(0, 1);
			double r = rand(mt);
			if(r > p){
				return 0;
			}
			return 1;
		}
		double uniform(double min, double max){
			std::uniform_real_distribution<double> rand(min, max);
			return rand(mt);
		}
		int uniform_int(int min, int max){
			std::uniform_int_distribution<> rand(min, max);
			return rand(mt);
		}
	}
}