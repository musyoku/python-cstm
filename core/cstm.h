#ifndef _cstm_
#define _cstm_
#include <unordered_map>
#include <vector>
#include <cassert>
#include "common.h"

class CSTM{
public:
	unordered_map<id, int> _n_k;				// 単語ベクトル
	unordered_map<id, double*> _word_vectors;	// 単語ベクトル
	vector<double*> _doc_vectors;				// 文書ベクトル
	int _ndim_d;
	int _num_documents;
	double _sigma_u;
	double _sigma_phi;
	double _sigma_alpha;
	double _alpha0;
	double* _tmp_vec;
	CSTM(){
		_ndim_d = NDIM_D;
		_num_documents = 0;
		_tmp_vec = generate_vector();
	}
	~CSTM(){
		for(auto &elem: _word_vectors){
			delete[] elem->second;
		}
		for(auto vec: _word_vectors){
			delete[] vec;
		}
	}
	void add_word(id word_id, id doc_id){
		assert(doc_id < _doc_vectors.size());
		auto itr = _word_vectors.find(word_id);
		if(itr != _word_vectors.end()){
			return;
		}
		double* word_vec = generate_vector();
		_word_vectors[word_id] = word_vec;
	}
	int add_document(){
		int doc_id = _num_documents;
		double* doc_vec = generate_vector();
		_doc_vectors.push_back(doc_vec);
		_num_documents++;
		return doc_id;
	}
	double* generate_vector(){
		double* vec = new double[_ndim_d];
		for(int i = 0;i < _ndim_d;i++){
			vec[i] = Sampler::normal(0, 1);
		}
		return vec;
	}
	double* draw_word_vec(double* old_vec){
		for(int i = 0;i < _ndim_d;i++){
			_tmp_vec[i] = Sampler::normal(old_vec[i], _sigma_phi);
		}
		return _tmp_vec;
	}
	double* draw_doc_vec(double* old_vec){
		for(int i = 0;i < _ndim_d;i++){
			_tmp_vec[i] = Sampler::normal(old_vec[i], _sigma_u);
		}
		return _tmp_vec;
	}
	double compute_alpha_word_given_doc(id word_id, double g0, int doc_id){
		auto itr = _word_vectors.find(word_id);
		assert(itr != _word_vectors.end());
		assert(doc_id < _doc_vectors.size());
		double* word_vec = itr->second;
		double* doc_vec = _doc_vectors[doc_id];
		double f = dot(word_vec, doc_vec, _ndim_d);
		return _alpha0 * g0 * exp(f);
	}
};

#endif