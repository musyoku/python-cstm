#pragma once
#include "fmath.h"

#define NDIM_D 			2		// 文書・単語ベクトルの次元数
#define SIGMA_U 		0.01	// 文書ベクトルのランダムウォーク幅
#define SIGMA_PHI 		0.02	// 単語ベクトルのランダムウォーク幅
#define SIGMA_ALPHA 	0.2		// a0のランダムウォーク幅
#define GAMMA_ALPHA_A 	5		// a0のガンマ事前分布のハイパーパラメータ
#define GAMMA_ALPHA_B 	1		// a0のガンマ事前分布のハイパーパラメータ
using id = size_t;

namespace cstm{
	double exp(double x){
		return fmath::expd(x);
		// return std::exp(x);
	}
	double dot(double* a, double* b, int length){
		double dot = 0;
		for(int i = 0;i < length;i++){
			dot += a[i] * b[i];
		}
		return dot;
	}
	void dump_vec(double* vec, int len){
		std::cout << "[";
		for(int i = 0;i < len - 1;i++){
			std::cout << vec[i] << ", ";
		}
		std::cout << vec[len - 1] << "]" << std::endl;
	}
}