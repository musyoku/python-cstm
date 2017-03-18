#pragma once
#include <vector>
#include <unordered_set>
#include <cassert>
#include <cmath>
#include "common.h"
#include "sampler.h"
using namespace std;
#define PI 3.14159265358979323846

class CSTM{
public:
	vector<hashmap<id, int>> _n_k;		// 文書ごとの単語の出現頻度
	vector<int> _sum_n_k;						// 文書ごとの単語の出現頻度の総和
	vector<double> _Zi;
	hashmap<id, double> _g0;				// 単語のデフォルト確率
	hashmap<id, double*> _word_vectors;	// 単語ベクトル
	vector<double*> _doc_vectors;				// 文書ベクトル
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
				hashmap<id, int> &count = _n_k[doc_id];
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
			hashmap<id, int> &count = _n_k[doc_id];
			for(const auto &elem: count){
				sum += elem.second;
			}
			_sum_n_k.push_back(sum);
		}
		assert(_n_k.size() == _num_documents);
		assert(_sum_n_k.size() == _num_documents);
		for(int doc_id = 0;doc_id < _num_documents;doc_id++){
			_Zi.push_back(0);
			update_Zi(doc_id);
		}
	}
	void add_word(id word_id, int doc_id){
		assert(doc_id < _doc_vectors.size());
		// カウントを更新
		hashmap<id, int> &count = _n_k[doc_id];
		count[word_id] += 1;
		_sum_word_frequency += 1;
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
		hashmap<id, int> count;
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
	double* draw_word_vector(double* old_vec){
		for(int i = 0;i < _ndim_d;i++){
			_tmp_vec[i] = old_vec[i] + Sampler::normal(0, _sigma_phi);
		}
		return _tmp_vec;
	}
	double* draw_doc_vector(double* old_vec){
		for(int i = 0;i < _ndim_d;i++){
			_tmp_vec[i] = old_vec[i] + Sampler::normal(0, _sigma_u);
		}
		return _tmp_vec;
	}
	void update_Zi(int doc_id){
		assert(doc_id < _Zi.size());
		_Zi[doc_id] = sum_alpha_word_given_doc(doc_id);
		// cout << "_Zi[" << doc_id << "] <- " << _Zi[doc_id] << endl;
	}
	double sum_alpha_word_given_doc(int doc_id){
		assert(doc_id < _n_k.size());
		double sum = 0;
		hashmap<id, int> &count = _n_k[doc_id];
		for(auto &elem: count){
			id word_id = elem.first;
			sum += compute_alpha_word_given_doc(word_id, doc_id);
		}
		return sum;
	}
	double compute_alpha_word_given_doc(id word_id, int doc_id){
		auto itr = _word_vectors.find(word_id);
		assert(itr != _word_vectors.end());
		assert(doc_id < _doc_vectors.size());
		double* word_vec = itr->second;
		double* doc_vec = _doc_vectors[doc_id];
		double f = std::dot(word_vec, doc_vec, _ndim_d);
		double g0 = get_g0_of_word(word_id);
		double alpha = _alpha0 * g0 * exp(f);
		return alpha;
	}
	double compute_reduced_log_Pdocument(id word_id, int doc_id){
		assert(doc_id < _sum_n_k.size());
		assert(doc_id < _n_k.size());
		assert(doc_id < _Zi.size());
		double log_pw = 0;
		double sum_alpha = _Zi[doc_id];
		double sum_word_frequency = _sum_n_k[doc_id];
		log_pw += lgamma(sum_alpha) - lgamma(sum_alpha + sum_word_frequency);
		double alpha_k = compute_alpha_word_given_doc(word_id, doc_id);
		int n_k = get_word_count_in_doc(word_id, doc_id);
		log_pw += lgamma(alpha_k + n_k) - lgamma(alpha_k);
		return log_pw;
	}
	double compute_log_Pdocument(unordered_set<id> &word_set, int doc_id){
		assert(doc_id < _sum_n_k.size());
		assert(doc_id < _n_k.size());
		assert(doc_id < _Zi.size());
		double log_pw = 0;
		double sum_alpha = _Zi[doc_id];
		// 
		// 
		// 
		// 
		// 
		// 
		// double _sum_alpha = sum_alpha_word_given_doc(doc_id);
		// if(abs(sum_alpha - _sum_alpha) > 1e-6){
		// 	printf("%.16e\n", sum_alpha);
		// 	printf("%.16e\n", _sum_alpha);
		// 	printf("%.16e\n", sum_alpha - _sum_alpha);
		// }
		// assert(abs(sum_alpha - _sum_alpha) < 1e-6);
		// 
		// 
		// 
		// 
		// 
		// 
		double sum_word_frequency = _sum_n_k[doc_id];
		// cout << "	" << "sum_alpha: " << sum_alpha << endl;
		// cout << "	" << "sum_word_frequency: " << sum_word_frequency << endl;
		log_pw += lgamma(sum_alpha) - lgamma(sum_alpha + sum_word_frequency);
		// if(std::isnan(log_pw)){
		// 	cout << sum_alpha << endl;
		// 	cout << sum_word_frequency << endl;
		// 	cout << lgamma(sum_alpha) << endl;
		// 	cout << lgamma(sum_alpha + sum_word_frequency) << endl;
		// 	exit(0);
		// }
		// int sum_n_k_check = 0;
		// double sum_alpha_check = 0;
		for(const id word_id: word_set){
			double alpha_k = compute_alpha_word_given_doc(word_id, doc_id);
			int n_k = get_word_count_in_doc(word_id, doc_id);
			// cout << "	" << word_id << endl;
			// cout << "	" << "alpha_k: " << alpha_k << endl;
			// cout << "	" << "n_k: " << n_k << endl;
			// cout << "	";
			// dump_vec(_word_vectors[word_id], _ndim_d);
			log_pw += lgamma(alpha_k + n_k) - lgamma(alpha_k);
			// sum_n_k_check += n_k;
			// sum_alpha_check += alpha_k;
		}
		//
		//
		//
		//
		//
		//
		//
		// if(abs(sum_alpha_check - sum_alpha) > 1e-6){
		// 	printf("%.16e\n", sum_alpha_check - sum_alpha);
		// }
		// assert(abs(sum_alpha_check - sum_alpha) < 1e-6);
		// if(abs(sum_n_k_check - sum_word_frequency) > 1e-6){
		// 	printf("%.16e\n", sum_n_k_check - sum_word_frequency);
		// }
		// assert(abs(sum_n_k_check - sum_word_frequency) < 1e-6);
		//
		//
		//
		//
		//
		//
		//
		// cout << "	" << "pw: " << exp(log_pw) << endl;
		return log_pw;
	}
	double compute_log_Pvector_doc(double* new_vec, double* old_vec){
		return _compute_log_Pvector_given_sigma(new_vec, old_vec, _sigma_u);
	}
	double compute_log_Pvector_word(double* new_vec, double* old_vec){
		return _compute_log_Pvector_given_sigma(new_vec, old_vec, _sigma_phi);
	}
	double _compute_log_Pvector_given_sigma(double* new_vec, double* old_vec, double sigma){
		double log_pvec = (double)_ndim_d * log(1.0 / (sqrt(2.0 * PI) * sigma));
		for(int i = 0;i < _ndim_d;i++){
			log_pvec -= (new_vec[i] - old_vec[i]) * (new_vec[i] - old_vec[i]) / (2.0 * sigma * sigma);		
		}
		return log_pvec;
	}
	double compute_log_prior_vector(double* vec){
		return _compute_log_prior_vector(vec);
	}
	double _compute_log_prior_vector(double* new_vec){
		double log_pvec = (double)_ndim_d * log(1.0 / (sqrt(2.0 * PI)));
		for(int i = 0;i < _ndim_d;i++){
			log_pvec -= new_vec[i] * new_vec[i] * 0.5;
		}
		return log_pvec;
	}
	double get_g0_of_word(id word_id){
		auto itr = _g0.find(word_id);
		assert(itr != _g0.end());
		return itr->second;
	}
	int get_sum_word_frequency_of_doc(int doc_id){
		assert(doc_id < _sum_n_k.size());
		return _sum_n_k[doc_id];
	}
	double* get_doc_vector(int doc_id){
		assert(doc_id < _doc_vectors.size());
		return _doc_vectors[doc_id];
	}
	double* get_word_vector(id word_id){
		auto itr = _word_vectors.find(word_id);
		assert(itr != _word_vectors.end());
		return itr->second;
	}
	int get_word_count_in_doc(id word_id, int doc_id){
		assert(doc_id < _n_k.size());
		hashmap<id, int> &count = _n_k[doc_id];
		auto itr = count.find(word_id);
		assert(itr != count.end());
		return itr->second;
	}
	double get_Zi(int doc_id){
		assert(doc_id < _Zi.size());
		return _Zi[doc_id];
	}
	void set_word_vector(id word_id, double* source){
		auto itr = _word_vectors.find(word_id);
		assert(itr != _word_vectors.end());
		double* target = itr->second;
		std::memcpy(target, source, _ndim_d * sizeof(double));
		// for(int i = 0;i < _ndim_d;i++){
		// 	target[i] = source[i];
		// }
	}
	void set_doc_vector(int doc_id, double* source){
		assert(doc_id < _doc_vectors.size());
		double* target = _doc_vectors[doc_id];
		std::memcpy(target, source, _ndim_d * sizeof(double));
		// for(int i = 0;i < _ndim_d;i++){
		// 	target[i] = source[i];
		// }
	}
	void swap_Zi_component(int doc_id, double old_value, double new_value){
		assert(doc_id < _Zi.size());
		_Zi[doc_id] += new_value - old_value;
	}
	void set_Zi(int doc_id, double new_value){
		assert(doc_id < _Zi.size());
		_Zi[doc_id] = new_value;
	}
};