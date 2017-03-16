#ifndef _cstm_
#define _cstm_
#include <unordered_map>
#include <vector>
#include <set>
#include <cassert>
#include <cmath>
#include "common.h"
#include "sampler.h"
using namespace std;
#define PI 3.14159265358979323846

class CSTM{
public:
	vector<unordered_map<id, int>> _n_k;		// 文書ごとの単語の出現頻度
	vector<double> _sum_n_k;					// 文書ごとの単語の出現頻度の総和
	unordered_map<id, double> _g0;				// 単語のデフォルト確率
	unordered_map<id, double*> _word_vectors;	// 単語ベクトル
	vector<double*> _doc_vectors;				// 文書ベクトル
	unordered_map<id, set<int>> _docs_containing_word;	// ある単語を含んでいる文書nのリスト
	int _ndim_d;
	int _num_documents;
	int _sum_word_frequency;	// 全単語の出現回数の総和
	double _sigma_u;
	double _sigma_phi;
	double _sigma_alpha;
	double _alpha0;
	double* _tmp_vec;
	CSTM(){
		_ndim_d = NDIM_D;
		_sigma_u = SIGMA_U;
		_sigma_phi = SIGMA_PHI;
		_sigma_alpha = SIGMA_ALPHA;
		_alpha0 = 1;
		_num_documents = 0;
		_sum_word_frequency = 0;
		_tmp_vec = generate_vector();
	}
	~CSTM(){
		for(auto &elem: _word_vectors){
			delete[] elem.second;
		}
		for(auto vec: _doc_vectors){
			delete[] vec;
		}
		delete[] _tmp_vec;
	}
	void compile(){
		for(const auto &elem: _word_vectors){
			id word_id = elem.first;
			double sum_count = 0;
			for(int doc_id = 0;doc_id < _num_documents;doc_id++){
				unordered_map<id, int> &count = _n_k[doc_id];
				auto itr = count.find(word_id);
				if(itr == count.end()){
					continue;
				}
				sum_count += itr->second;
			}
			double g0 = sum_count / _sum_word_frequency;
			_g0[word_id] = g0;
		}
		for(int doc_id = 0;doc_id < _num_documents;doc_id++){
			int sum = 0;
			unordered_map<id, int> &count = _n_k[doc_id];
			for(const auto &elem: count){
				sum += elem.second;
			}
			_sum_n_k.push_back(sum);
		}
		assert(_n_k.size() == _num_documents);
		assert(_sum_n_k.size() == _num_documents);
	}
	void add_word(id word_id, int doc_id){
		assert(doc_id < _doc_vectors.size());
		// カウントを更新
		unordered_map<id, int> &count = _n_k[doc_id];
		count[word_id] += 1;
		_sum_word_frequency += 1;
		// 文書との対応
		set<int> &docs = _docs_containing_word[word_id];
		docs.insert(doc_id);
		// 単語ベクトルを必要なら生成
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
		unordered_map<id, int> count;
		_n_k.push_back(count);
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
	double compute_alpha_word_given_doc(id word_id, int doc_id){
		auto itr = _word_vectors.find(word_id);
		assert(itr != _word_vectors.end());
		assert(doc_id < _doc_vectors.size());
		double* word_vec = itr->second;
		double* doc_vec = _doc_vectors[doc_id];
		double f = std::dot(word_vec, doc_vec, _ndim_d);
		double g0 = get_g0_for_word(word_id);
		return _alpha0 * g0 * exp(f);
	}
	double sum_alpha_doc(int doc_id){
		assert(doc_id < _n_k.size());
		double sum = 0;
		unordered_map<id, int> &count = _n_k[doc_id];
		for(auto &elem: count){
			id word_id = elem.first;
			sum += compute_alpha_word_given_doc(word_id, doc_id);
		}
		return sum;
	}
	double compute_Pw_given_doc(int doc_id){
		assert(doc_id < _n_k.size());
		double log_pw = 0;
		double sum_alpha = sum_alpha_doc(doc_id);
		double sum_word_frequency = get_sum_word_frequency_of_doc(doc_id);
		log_pw += lgamma(sum_alpha) - lgamma(sum_alpha + sum_word_frequency);
		unordered_map<id, int> &count = _n_k[doc_id];
		for(const auto &elem: count){
			id word_id = elem.first;
			double alpha_k = compute_alpha_word_given_doc(word_id, doc_id);
			int n_k = elem.second;
			log_pw += lgamma(alpha_k + n_k) - lgamma(alpha_k);
		}
		return log_pw;
	}
	double compute_log_Pvec_doc(double* new_vec, double* old_vec){
		return _compute_log_Pvec_given_sigma(new_vec, old_vec, _sigma_u);
	}
	double compute_log_Pvec_word(double* new_vec, double* old_vec){
		return _compute_log_Pvec_given_sigma(new_vec, old_vec, _sigma_phi);
	}
	double _compute_log_Pvec_given_sigma(double* new_vec, double* old_vec, double sigma){
		double log_pvec = (double)_ndim_d * log(1.0 / (sqrt(2.0 * PI) * sigma));
		for(int i = 0;i < _ndim_d;i++){
			log_pvec -= (new_vec[i] - old_vec[i]) * (new_vec[i] - old_vec[i]) / (2.0 * sigma * sigma);		
		}
		return log_pvec;
	}
	double get_g0_for_word(id word_id){
		auto itr = _g0.find(word_id);
		assert(itr != _g0.end());
		return itr->second;
	}
	int get_sum_word_frequency_of_doc(int doc_id){
		assert(doc_id < _sum_n_k.size());
		return _sum_n_k[doc_id];
	}
	double* get_doc_vec(int doc_id){
		assert(doc_id < _doc_vectors.size());
		return _doc_vectors[doc_id];
	}
	double* get_word_vec(id word_id){
		auto itr = _word_vectors.find(word_id);
		assert(itr != _word_vectors.end());
		return itr->second;
	}
	void set_word_vector(id word_id, double* source){
		auto itr = _word_vectors.find(word_id);
		assert(itr != _word_vectors.end());
		double* target = itr->second;
		for(int i = 0;i < _ndim_d;i++){
			target[i] = source[i];
		}
	}
	void set_doc_vector(int doc_id, double* source){
		assert(doc_id < _doc_vectors.size());
		double* target = _doc_vectors[doc_id];
		for(int i = 0;i < _ndim_d;i++){
			target[i] = source[i];
		}
	}
};

#endif