#include <iostream>
#include <cmath>
#include "common.h"

namespace cstm {
	double exp(double x){
		return std::exp(x);
	}
	double norm(double* a, int length){
		double norm = 0;
		for(int i = 0;i < length;i++){
			norm += a[i] * a[i];
		}
		return sqrt(norm);
	}
	double inner(double* a, double* b, int length){
		double inner = 0;
		for(int i = 0;i < length;i++){
			inner += a[i] * b[i];
		}
		return inner;
	}
	void dump_vec(double* vec, int len){
		std::cout << "[";
		for(int i = 0;i < len - 1;i++){
			std::cout << vec[i] << ", ";
		}
		std::cout << vec[len - 1] << "]" << std::endl;
	}
}