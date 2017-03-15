#ifndef _cstm_
#define _cstm_
#include <unordered_map>
#include "common.h"

class CSTM{
public:
	unordered_map<id, double*> _word_vectors;
	unordered_map<int, double*> _doc_vectors;
	int _ndim_d;
	CSTM(){
		_ndim_d = NDIM_D;
	}
	void add_word(id word_id){

	}
	void add_document(int doc_id){

	}
	double* generate_vector(){
		double* vec = new double[_ndim_d];
		for(int i = 0;i < _ndim_d;i++){
			vec[i] = Sampler::normal(0, 1);
		}
		return vec;
	}
};

#endif