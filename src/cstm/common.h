#pragma once
#include <cstddef>

#define NDIM_D 			2		// 文書・単語ベクトルの次元数
#define SIGMA_U 		0.01	// 文書ベクトルのランダムウォーク幅
#define SIGMA_PHI 		0.02	// 単語ベクトルのランダムウォーク幅
#define SIGMA_ALPHA 	0.2		// a0のランダムウォーク幅
#define GAMMA_ALPHA_A 	5		// a0のガンマ事前分布のハイパーパラメータ
#define GAMMA_ALPHA_B 	1		// a0のガンマ事前分布のハイパーパラメータ

using id = size_t;
#define SPECIAL_WORD_UNK 0

namespace cstm {
	double exp(double x);
	double norm(double* a, int length);
	double inner(double* a, double* b, int length);
	void dump_vec(double* vec, int len);
}