#pragma once
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
// #include <boost/serialization/split_free.hpp>
#include <boost/serialization/split_member.hpp>
#include <unordered_set>
#include <cassert>
#include <cmath>
#include <random>
#include <fstream>
#include "common.h"
#include "sampler.h"
#define PI 3.14159265358979323846	// 直書き

namespace cstm{
	class CSTM{
	private:
		friend class boost::serialization::access;
		template <class Archive>
		void serialize(Archive &archive, unsigned int version);
		void save(boost::archive::binary_oarchive &archive, unsigned int version) const;
		void load(boost::archive::binary_iarchive &archive, unsigned int version);
	public:
		int** _n_k;					// 文書ごとの単語の出現頻度
		int* _sum_n_k;				// 文書ごとの単語の出現頻度の総和
		int* _word_count;
		double* _Zi;
		double* _g0;				// 単語のデフォルト確率
		double** _word_vectors;		// 単語ベクトル
		double** _doc_vectors;		// 文書ベクトル
		int _ndim_d;
		int _num_documents;
		int _vocabulary_size;
		int _sum_word_frequency;	// 全単語の出現回数の総和
		int _ignore_word_count;
		double _sigma_u;
		double _sigma_phi;
		double _sigma_alpha0;
		double _gamma_alpha_a;
		double _gamma_alpha_b;
		double _alpha0;
		double* _tmp_vec;
		double* _log_likelihood_first_term;
		std::normal_distribution<double> _standard_normal_distribution;
		std::normal_distribution<double> _noise_word;
		std::normal_distribution<double> _noise_doc;
		std::normal_distribution<double> _noise_alpha0;
		CSTM(int ndim_d, int vocabulary_size, int num_documents);
		~CSTM();
		void _allocate_capacity(int ndim_d, int vocabulary_size, int num_documents);
		void _delete_capacity();
		void add_word(id word_id, int doc_id);
		double generate_noise_from_standard_normal_distribution();
		double generate_noise_doc();
		double generate_noise_word();
		double* generate_vector();
		double* draw_word_vector(double* old_vec);
		double* draw_doc_vector(double* old_vec);
		double draw_alpha0(double old_alpha0);
		double sum_alpha_word_given_doc(int doc_id);
		double compute_alpha_word_given_doc(id word_id, int doc_id);
		double _compute_alpha_word(double* word_vec, double* doc_vec, double g0);
		double compute_reduced_log_probability_document(id word_id, int doc_id);
		double _compute_reduced_log_probability_document(id word_id, int doc_id, int n_k, double Zi, double alpha_k);
		double compute_log_probability_document(int doc_id);
		double compute_log_probability_document_given_words(int doc_id, std::unordered_set<id> &word_ids);
		double _compute_second_term_of_log_probability_document(int doc_id, id word_id);
		double compute_log_prior_alpha0(double alpha0);
		double compute_log_Pvector_doc(double* new_vec, double* old_vec);
		double compute_log_Pvector_word(double* new_vec, double* old_vec);
		double _compute_log_Pvector_given_sigma(double* new_vec, double* old_vec, double sigma);
		double compute_log_prior_vector(double* vec);
		double get_alpha0();
		double get_g0_of_word(id word_id);
		int get_sum_word_frequency_of_doc(int doc_id);
		double* get_doc_vector(int doc_id);
		double* get_word_vector(id word_id);
		int get_word_count_in_doc(id word_id, int doc_id);
		int get_word_count(id word_id);
		int get_ignore_word_count();
		double get_Zi(int doc_id);
		void set_ndim_d(int ndim_d);
		void set_alpha0(double alpha0);
		void set_sigma_u(double sigma_u);
		void set_sigma_phi(double sigma_phi);
		void set_sigma_alpha(double sigma_alpha);
		void set_gamma_alpha_a(double gamma_alpha_a);
		void set_gamma_alpha_b(double gamma_alpha_b);
		void set_num_documents(int num_documents);
		void set_size_vocabulary(int vocabulary_size);
		void set_ignore_word_count(int count);
		void set_word_vector(id word_id, double* source);
		void set_doc_vector(int doc_id, double* source);
		void update_Zi(int doc_id);
		void set_Zi(int doc_id, double new_value);
		// void save(std::string filename);
		// bool load(std::string filename);
	};
}
// モデルの保存用
// namespace boost { namespace serialization {
// template<class Archive>
// 	void save(Archive &archive, const cstm::CSTM &cstm, unsigned int version);
// template<class Archive>
// 	void load(Archive &archive, cstm::CSTM &cstm, unsigned int version);
// }}