#pragma once
#include <iostream>
#include <unordered_map>
#include "hashmap.h"
using namespace std;
using namespace emilib;
template<class T, class U>
// using hashmap = unordered_map<T, U>;
using hashmap = HashMap<T, U>;

#define NDIM_D 		3		// 文書・単語ベクトルの次元数
#define SIGMA_U 	0.01	// 文書ベクトルのランダムウォーク幅
#define SIGMA_PHI 	0.02	// 単語ベクトルのランダムウォーク幅
#define SIGMA_ALPHA 0.2		// a0のランダムウォーク幅

using id = size_t;

double compute_dot(double* a, double* b, int length){
	double dot = 0;
	for(int i = 0;i < length;i++){
		dot += a[i] * b[i];
	}
	return dot;
}
void dump_vec(double* vec, int len){
	cout << "[";
	for(int i = 0;i < len - 1;i++){
		cout << vec[i] << ", ";
	}
	cout << vec[len - 1] << "]" << endl;
}