#pragma once
#include <unordered_set>
#include <cassert>
#include <cmath>
#include <random>
#include "common.h"
#include "sampler.h"
using namespace std;
#define PI 3.14159265358979323846

class CSTM{
public:
	int** _n_k;					// 文書ごとの単語の出現頻度
	int* _sum_n_k;				// 文書ごとの単語の出現頻度の総和
	double* _Zi;
	double* _g0;				// 単語のデフォルト確率
	double** _word_vectors;		// 単語ベクトル
	double** _doc_vectors;		// 文書ベクトル
	int _ndim_d;
	int _num_documents;
	int _num_vocabulary;
	int _sum_word_frequency;	// 全単語の出現回数の総和
	double _sigma_u;
	double _sigma_phi;
	double _sigma_alpha;
	double _alpha0;
	double* _tmp_vec;
	normal_distribution<double> standard_normal_distribution;
	normal_distribution<double> noise_word;
	normal_distribution<double> noise_doc;
	CSTM(int num_documents, int num_vocabulary){
		_ndim_d = NDIM_D;
		_sigma_u = SIGMA_U;
		_sigma_phi = SIGMA_PHI;
		_sigma_alpha = SIGMA_ALPHA;
		_num_vocabulary = num_vocabulary;
		_num_documents = num_documents;
		standard_normal_distribution = normal_distribution<double>(0, 1);
		noise_word = normal_distribution<double>(0, _sigma_u);
		noise_doc = normal_distribution<double>(0, _sigma_phi);
		_alpha0 = 1;
		_sum_word_frequency = 0;
		_tmp_vec = generate_vector();
		_g0 = new double[_num_vocabulary];
		_word_vectors = new double*[_num_vocabulary];
		_doc_vectors = new double*[_num_documents];
		_n_k = new int*[_num_documents];
		_sum_n_k = new int[_num_documents];
		_Zi = new double[_num_documents];
		for(id word_id = 0;word_id < _num_vocabulary;word_id++){
			_word_vectors[word_id] = NULL;
		}
		for(int doc_id = 0;doc_id < _num_documents;doc_id++){
			_doc_vectors[doc_id] = NULL;
			_n_k[doc_id] = new int[_num_vocabulary];
			_Zi[doc_id] = 0;
			for(id word_id = 0;word_id < _num_vocabulary;word_id++){
				_n_k[doc_id][word_id] = 0;
			}
		}
	}
	~CSTM(){
		for(id word_id = 0;word_id < _num_vocabulary;word_id++){
			if(_word_vectors[word_id] != NULL){
				delete[] _word_vectors[word_id];
			}
		}
		for(int doc_id = 0;doc_id < _num_documents;doc_id++){
			delete[] _doc_vectors[doc_id];
		}
		delete[] _tmp_vec;
	}
	void compile(){
		for(id word_id = 0;word_id < _num_vocabulary;word_id++){
			double sum_count = 0;
			for(int doc_id = 0;doc_id < _num_documents;doc_id++){
				int* count = _n_k[doc_id];
				assert(count[word_id] >= 0);
				sum_count += count[word_id];
			}
			double g0 = sum_count / _sum_word_frequency;
			assert(0 < g0 && g0 <= 1);
			_g0[word_id] = g0;
		}
		int sum_word_frequency_check = 0;
		for(int doc_id = 0;doc_id < _num_documents;doc_id++){
			int sum = 0;
			int* count = _n_k[doc_id];
			for(id word_id = 0;word_id < _num_vocabulary;word_id++){
				sum += count[word_id];
			}
			_sum_n_k[doc_id] = sum;
			sum_word_frequency_check += sum;
		}
		assert(sum_word_frequency_check == _sum_word_frequency);
	}
	void add_word(id word_id, int doc_id){
		assert(doc_id < _num_documents);
		assert(word_id < _num_vocabulary);
		// カウントを更新
		int* count = _n_k[doc_id];
		count[word_id] += 1;
		_sum_word_frequency += 1;
		// 単語ベクトルを必要なら生成
		if(_word_vectors[word_id] != NULL){
			return;
		}
		double* word_vec = generate_vector();
		_word_vectors[word_id] = word_vec;
	}
	void add_document(int doc_id){
		assert(doc_id < _num_documents);
		if(_doc_vectors[doc_id] != NULL){
			return;
		}
		double* doc_vec = generate_vector();
		_doc_vectors[doc_id] = doc_vec;
	}
	double generate_noise_from_standard_normal_distribution(){
		return standard_normal_distribution(Sampler::minstd);
	}
	double generate_noise_doc(){
		return noise_doc(Sampler::minstd);
	}
	double generate_noise_word(){
		return noise_word(Sampler::minstd);
	}
	double* generate_vector(){
		double* vec = new double[_ndim_d];
		for(int i = 0;i < _ndim_d;i++){
			vec[i] = generate_noise_from_standard_normal_distribution();
		}
		return vec;
	}
	double* draw_word_vector(double* old_vec){
		for(int i = 0;i < _ndim_d;i++){
			_tmp_vec[i] = old_vec[i] + generate_noise_word();
		}
		return _tmp_vec;
	}
	double* draw_doc_vector(double* old_vec){
		for(int i = 0;i < _ndim_d;i++){
			_tmp_vec[i] = old_vec[i] + generate_noise_doc();
		}
		return _tmp_vec;
	}
	void update_Zi(int doc_id, unordered_set<id> &word_set){
		assert(doc_id < _num_documents);
		_Zi[doc_id] = sum_alpha_word_given_doc(doc_id, word_set);
		// cout << "_Zi[" << doc_id << "] <- " << _Zi[doc_id] << endl;
	}
	double sum_alpha_word_given_doc(int doc_id, unordered_set<id> &word_set){
		assert(doc_id < _num_documents);
		double sum = 0;
		for(const id word_id: word_set){
			sum += compute_alpha_word_given_doc(word_id, doc_id);
		}
		return sum;
	}
	double compute_alpha_word_given_doc(id word_id, int doc_id){
		assert(word_id < _num_vocabulary);
		assert(doc_id < _num_documents);
		double* word_vec = _word_vectors[word_id];
		double* doc_vec = _doc_vectors[doc_id];
		double f = compute_dot(word_vec, doc_vec, _ndim_d);
		double g0 = get_g0_of_word(word_id);
		double alpha = _alpha0 * g0 * exp(f);
		assert(alpha > 0);
		return alpha;
	}
	double compute_reduced_log_Pdocument(id word_id, int doc_id){
		assert(doc_id < _num_documents);
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
		assert(doc_id < _num_documents);
		double log_pw = 0;
		double sum_alpha = _Zi[doc_id];
		// printf("%.16e\n", sum_alpha);
		assert(sum_alpha > 0);
		// 
		// 
		// 
		// 
		// 
		// 
		// double _sum_alpha = sum_alpha_word_given_doc(doc_id, word_set);
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
		// 	printf("%.16e\n", sum_alpha_check);
		// 	printf("%.16e\n", sum_alpha);
		// 	printf("%.16e\n", abs(sum_alpha_check - sum_alpha));
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
		assert(word_id < _num_vocabulary);
		return _g0[word_id];
	}
	int get_sum_word_frequency_of_doc(int doc_id){
		assert(doc_id < _num_documents);
		return _sum_n_k[doc_id];
	}
	double* get_doc_vector(int doc_id){
		assert(doc_id < _num_documents);
		return _doc_vectors[doc_id];
	}
	double* get_word_vector(id word_id){
		assert(word_id < _num_vocabulary);
		return _word_vectors[word_id];
	}
	int get_word_count_in_doc(id word_id, int doc_id){
		assert(doc_id < _num_documents);
		assert(word_id < _num_vocabulary);
		int* count = _n_k[doc_id];
		return count[word_id];
	}
	double get_Zi(int doc_id){
		assert(doc_id < _num_documents);
		return _Zi[doc_id];
	}
	void set_word_vector(id word_id, double* source){
		assert(word_id < _num_vocabulary);
		double* target = _word_vectors[word_id];
		std::memcpy(target, source, _ndim_d * sizeof(double));
		// for(int i = 0;i < _ndim_d;i++){
		// 	target[i] = source[i];
		// }
	}
	void set_doc_vector(int doc_id, double* source){
		assert(doc_id < _num_documents);
		double* target = _doc_vectors[doc_id];
		std::memcpy(target, source, _ndim_d * sizeof(double));
		// for(int i = 0;i < _ndim_d;i++){
		// 	target[i] = source[i];
		// }
	}
	void swap_Zi_component(int doc_id, double old_value, double new_value){
		assert(doc_id < _num_documents);
		_Zi[doc_id] += new_value - old_value;
	}
	void set_Zi(int doc_id, double new_value){
		assert(doc_id < _num_documents);
		_Zi[doc_id] = new_value;
	}
};