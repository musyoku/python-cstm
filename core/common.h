#ifndef _common_
#define _common_

#define NDIM_D 		2		// 文書・単語ベクトルの次元数
#define SIGMA_U 	0.01	// 文書ベクトルのランダムウォーク幅
#define SIGMA_PHI 	0.02	// 単語ベクトルのランダムウォーク幅
#define SIGMA_ALPHA 0.2		// a0のランダムウォーク幅

using id = size_t;
#define ID_BOS 0
#define ID_EOS 1

double dot(double* a, double* b, int length){
	double dot = 0;
	for(int i = 0;i < length;i++){
		dot += a[i] * b[i];
	}
	return dot;
}
#endif