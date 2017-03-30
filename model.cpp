#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_set.hpp>
#include <boost/python.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <string>
#include <set>
#include <unordered_set>
#include <unordered_map> 
#include "core/fmath.h"
#include "core/cstm.h"
#include "core/vocab.h"
using namespace boost;
using namespace cstm;

struct multiset_value_comparator {
	bool operator()(const pair<id, double> &a, const pair<id, double> &b) {
		return a.second > b.second;
	}   
};

void split_word_by(const wstring &str, wchar_t delim, vector<wstring> &elems){
	elems.clear();
	wstring item;
	for(wchar_t ch: str){
		if (ch == delim){
			if (!item.empty()){
				elems.push_back(item);
			}
			item.clear();
		}
		else{
			item += ch;
		}
	}
	if (!item.empty()){
		elems.push_back(item);
	}
}

template<class T>
python::list list_from_vector(vector<T> &vec){  
	 python::list list;
	 typename vector<T>::const_iterator it;

	 for(it = vec.begin(); it != vec.end(); ++it)   {
		  list.append(*it);
	 }
	 return list;
}

class PyCSTM{
public:
	CSTM* _cstm;
	Vocab* _vocab;
	vector<vector<vector<id>>> _dataset;
	vector<unordered_set<id>> _word_ids_in_doc;
	vector<int> _sum_word_frequency;	// 文書ごとの単語の出現頻度の総和
	vector<id> _random_word_ids;
	vector<int> _random_doc_ids;
	unordered_map<id, unordered_set<int>> _docs_containing_word;	// ある単語を含んでいる文書nのリスト
	unordered_map<id, int> _word_frequency;
	double* _old_vec_copy;
	double* _new_vec_copy;
	double* _old_alpha_words;
	double* _Zi_cache;
	bool _is_compiled;
	int _ndim_d;
	// 統計
	// MH法で採択された回数
	int _num_acceptance_doc;
	int _num_acceptance_word;
	int _num_acceptance_alpha0;
	// MH法で棄却された回数
	int _num_rejection_doc;
	int _num_rejection_word;
	int _num_rejection_alpha0;
	// サンプリング回数
	int _num_word_vec_sampled;
	int _num_doc_vec_sampled;
	// その他
	int _random_sampling_word_index;
	int _random_sampling_doc_index;
	unordered_map<id, int> _num_updates_word;
	unordered_map<int, int> _num_updates_doc;

	PyCSTM(){
		setlocale(LC_CTYPE, "ja_JP.UTF-8");
		ios_base::sync_with_stdio(false);
		locale default_loc("ja_JP.UTF-8");
		locale::global(default_loc);
		locale ctype_default(locale::classic(), default_loc, locale::ctype); //※
		wcout.imbue(ctype_default);
		wcin.imbue(ctype_default);
		_cstm = new CSTM();
		_vocab = new Vocab();
		_old_vec_copy = NULL;
		_new_vec_copy = NULL;
		_old_alpha_words = NULL;
		_Zi_cache = NULL;
		_ndim_d = 0;
		reset_statistics();
		_is_compiled = false;
		_random_sampling_word_index = 0;
		_random_sampling_doc_index = 0;
	}
	~PyCSTM(){
		delete _cstm;
		delete _vocab;
		if(_old_vec_copy != NULL){
			delete[] _old_vec_copy;
		}
		if(_new_vec_copy != NULL){
			delete[] _new_vec_copy;
		}
		if(_old_alpha_words != NULL){
			delete[] _old_alpha_words;
		}
		if(_Zi_cache != NULL){
			delete[] _Zi_cache;
		}
	}
	void compile_if_needed(){
		if(_is_compiled){
			return;
		}
		compile();
	}
	void compile(){
		assert(_ndim_d > 0);
		assert(_is_compiled == false);
		int num_docs = _dataset.size();
		int num_vocabulary = _word_frequency.size();
		_old_vec_copy = new double[_ndim_d];
		_new_vec_copy = new double[_ndim_d];
		// 単語のランダムサンプリング用
		for(id word_id = 0;word_id < num_vocabulary;word_id++){
			_random_word_ids.push_back(word_id);
			_num_updates_word[word_id] = 0;
		}
		// CSTM
		_cstm->set_ndim_d(_ndim_d);
		_cstm->set_num_documents(num_docs);
		_cstm->set_num_vocabulary(num_vocabulary);
		_cstm->init();
		for(int doc_id = 0;doc_id < num_docs;doc_id++){
			vector<vector<id>> &dataset = _dataset[doc_id];
			for(int data_index = 0;data_index < dataset.size();data_index++){
				vector<id> &word_ids = dataset[data_index];
				for(const id word_id: word_ids){
					_cstm->add_word(word_id, doc_id);
				}
			}
			_num_updates_doc[doc_id] = 0;
			_random_doc_ids.push_back(doc_id);
		}
		_cstm->compile();
		assert(_ndim_d == _cstm->_ndim_d);
		// Zi
		for(int doc_id = 0;doc_id < num_docs;doc_id++){
			_cstm->update_Zi(doc_id);
		}
		assert(_sum_word_frequency.size() == _dataset.size());
		_old_alpha_words = new double[num_docs];
		_Zi_cache = new double[num_docs];
		std::shuffle(_random_word_ids.begin(), _random_word_ids.end(), Sampler::mt);
		std::shuffle(_random_doc_ids.begin(), _random_doc_ids.end(), Sampler::mt);
		_is_compiled = true;
	}
	int add_document(string filename){
		wifstream ifs(filename.c_str());
		assert(ifs.fail() == false);
		// 文書の追加
		int doc_id = _dataset.size();
		_dataset.push_back(vector<vector<id>>());
		_word_ids_in_doc.push_back(unordered_set<id>());
		_sum_word_frequency.push_back(0);
		// ファイルの読み込み
		wstring sentence;
		vector<wstring> sentences;
		while (getline(ifs, sentence) && !sentence.empty()){
			assert(PyErr_CheckSignals() == 0);	// ctrl+cが押されたかチェック
			sentences.push_back(sentence);
		}
		vector<int> rand_indices;
		for(int i = 0;i < sentences.size();i++){
			rand_indices.push_back(i);
		}
		shuffle(rand_indices.begin(), rand_indices.end(), Sampler::mt);	// データをシャッフル
		for(int i = 0;i < rand_indices.size();i++){
			wstring &sentence = sentences[rand_indices[i]];
			vector<wstring> words;
			split_word_by(sentence, L' ', words);	// スペースで分割
			add_sentence_to_doc(words, doc_id);
		}
		return doc_id;
	}
	void add_sentence_to_doc(vector<wstring> &words, int doc_id){
		if(words.size() > 0){
			assert(doc_id < _dataset.size());
			vector<vector<id>> &dataset = _dataset[doc_id];
			_sum_word_frequency[doc_id] += words.size();
			vector<id> word_ids;
			for(auto word: words){
				if(word.size() == 0){
					continue;
				}
				id word_id = _vocab->add_string(word);
				word_ids.push_back(word_id);
				unordered_set<int> &docs = _docs_containing_word[word_id];
				docs.insert(doc_id);
				unordered_set<id> &word_ids = _word_ids_in_doc[doc_id];
				word_ids.insert(word_id);
				_word_frequency[word_id] += 1;
			}
			dataset.push_back(word_ids);
		}
	}
	bool is_doc_contain_word(int doc_id, id word_id){
		unordered_set<int> &set = _docs_containing_word[word_id];
		auto itr = set.find(doc_id);
		return itr != set.end();
	}
	int get_num_documents(){
		return _cstm->_num_documents;
	}
	int get_num_vocabulary(){
		return _cstm->_num_vocabulary;
	}
	int get_ndim_d(){
		return _cstm->_ndim_d;
	}
	int get_sum_word_frequency(){
		return std::accumulate(_sum_word_frequency.begin(), _sum_word_frequency.end(), 0);
	}
	int get_num_word_vec_sampled(){
		return _num_word_vec_sampled;
	}
	int get_num_doc_vec_sampled(){
		return _num_doc_vec_sampled;
	}
	double get_alpha0(){
		return _cstm->_alpha0;
	}
	double get_mh_acceptance_rate_for_doc_vector(){
		return _num_acceptance_doc / (double)(_num_acceptance_doc + _num_rejection_doc);
	}
	double get_mh_acceptance_rate_for_word_vector(){
		return _num_acceptance_word / (double)(_num_acceptance_word + _num_rejection_word);
	}
	double get_mh_acceptance_rate_for_alpha0(){
		return _num_acceptance_alpha0 / (double)(_num_acceptance_alpha0 + _num_rejection_alpha0);
	}
	double* get_word_vector(id word_id){
		double* old_vec = _cstm->get_word_vector(word_id);
		std::memcpy(_old_vec_copy, old_vec, _cstm->_ndim_d * sizeof(double));
		return _old_vec_copy;
	}
	double* get_doc_vector(int doc_id){
		double* old_vec = _cstm->get_doc_vector(doc_id);
		std::memcpy(_old_vec_copy, old_vec, _cstm->_ndim_d * sizeof(double));
		return _old_vec_copy;
	}
	double* draw_word_vector(double* old_vec){
		double* new_vec = _cstm->draw_word_vector(old_vec);
		std::memcpy(_new_vec_copy, new_vec, _cstm->_ndim_d * sizeof(double));
		return _new_vec_copy;
	}
	double* draw_doc_vector(double* old_vec){
		double* new_vec = _cstm->draw_doc_vector(old_vec);
		std::memcpy(_new_vec_copy, new_vec, _cstm->_ndim_d * sizeof(double));
		return _new_vec_copy;
	}
	void set_ndim_d(int ndim_d){
		_ndim_d = ndim_d;
	}
	void set_alpha0(double alpha0){
		_cstm->_alpha0 = alpha0;
	}
	void set_sigma_u(double sigma_u){
		_cstm->_sigma_u = sigma_u;
	}
	void set_sigma_phi(double sigma_phi){
		_cstm->_sigma_phi = sigma_phi;
	}
	void set_sigma_alpha0(double sigma_alpha0){
		_cstm->_sigma_alpha0 = sigma_alpha0;
	}
	void set_gamma_alpha_a(double gamma_alpha_a){
		_cstm->_gamma_alpha_a = gamma_alpha_a;
	}
	void set_gamma_alpha_b(double gamma_alpha_b){
		_cstm->_gamma_alpha_b = gamma_alpha_b;
	}
	python::list convert_vector_to_list(double* vector){
		python::list vector_list;
		for(int i = 0;i < _cstm->_ndim_d;i++){
			vector_list.append(vector[i]);
		}
		return vector_list;
	}
	python::list get_word_vectors(){
		python::list vector_array;
		for(id word_id = 0;word_id < get_num_vocabulary();word_id++){
			python::list vector_list;
			double* vector = get_word_vector(word_id);
			vector_array.append(convert_vector_to_list(vector));
		}
		return vector_array;
	}
	python::list get_doc_vectors(){
		python::list vector_array;
		for(int doc_id = 0;doc_id < get_num_documents();doc_id++){
			python::list vector_list;
			double* vector = get_doc_vector(doc_id);
			for(int i = 0;i < _cstm->_ndim_d;i++){
				vector_list.append(vector[i]);
			}
			vector_array.append(vector_list);
		}
		return vector_array;
	}
	// 出現頻度が高い単語とベクトルのペアを返す
	python::list get_high_freq_words(size_t threshold = 100){
		python::list result;
		std::pair<id, double> pair;
		multiset<std::pair<id, double>, multiset_value_comparator> ranking;
		for(id word_id = 0;word_id < get_num_vocabulary();word_id++){
			int count = _word_frequency[word_id];
			pair.first = word_id;
			pair.second = count;
			ranking.insert(pair);
		}
		auto itr = ranking.begin();
		for(int n = 0;n < std::min(threshold, ranking.size());n++){
			python::list tuple;
			id word_id = itr->first;
			wstring word = _vocab->word_id_to_string(word_id);
			double* vector = get_word_vector(word_id);
			int count = itr->second;
			tuple.append(word_id);
			tuple.append(word);
			tuple.append(count);
			tuple.append(convert_vector_to_list(vector));
			result.append(tuple);
			itr++;
		}
		return result;
	}
	void reset_statistics(){
		_num_acceptance_doc = 0;
		_num_acceptance_word = 0;
		_num_acceptance_alpha0 = 0;
		_num_rejection_doc = 0;
		_num_rejection_word = 0;
		_num_rejection_alpha0 = 0;
		_num_word_vec_sampled = 0;
		_num_doc_vec_sampled = 0;
	}
	double compute_log_likelihood_data(){
		double log_pw = 0;
		int n = 0;
		for(int doc_id = 0;doc_id < get_num_documents();doc_id++){
			unordered_set<id> &word_ids = _word_ids_in_doc[doc_id];
			log_pw += _cstm->compute_log_probability_document_given_words(doc_id, word_ids);
		}
		return log_pw;
	}
	double compute_perplexity(){
		double log_pw = 0;
		int n = 0;
		for(int doc_id = 0;doc_id < get_num_documents();doc_id++){
			unordered_set<id> &word_ids = _word_ids_in_doc[doc_id];
			log_pw += _cstm->compute_log_probability_document_given_words(doc_id, word_ids);
		}
		return fmath::expd(-log_pw / get_sum_word_frequency());
	}
	void update_all_Zi(){
		for(int doc_id = 0;doc_id < get_num_documents();doc_id++){
			_cstm->update_Zi(doc_id);
		}
	}
	void perform_mh_sampling_word(){
		compile_if_needed();
		// 更新する単語ベクトルをランダムに選択
		// 一度に更新する個数は 語彙数/文書数
		int limit = (int)(get_num_vocabulary() / (double)get_num_documents());
		if(_random_sampling_word_index + limit >= _random_word_ids.size()){
			std::shuffle(_random_word_ids.begin(), _random_word_ids.end(), Sampler::mt);
			_random_sampling_word_index = 0;
		}
		for(int i = 0;i < limit;i++){
			id word_id = _random_word_ids[i + _random_sampling_word_index];
			double* old_vec = get_word_vector(word_id);
			double* new_vec = draw_word_vector(old_vec);
			accept_word_vecor_if_needed(new_vec, old_vec, word_id);
			_num_word_vec_sampled += 1;
			_num_updates_word[word_id] += 1;
		}
		_random_sampling_word_index += limit;
	}
	bool accept_word_vecor_if_needed(double* new_word_vec, double* old_word_vec, id word_id){
		auto itr = _docs_containing_word.find(word_id);
		assert(itr != _docs_containing_word.end());
		unordered_set<int> &docs = itr->second;
		assert(docs.size() > 0);
		// 以前の単語ベクトルの尤度
		double log_pw_old = 0;
		for(int doc_id = 0;doc_id < get_num_documents();doc_id++){
			double old_alpha_word = _cstm->compute_alpha_word_given_doc(word_id, doc_id);
			double old_Zi = _cstm->get_Zi(doc_id);
			int n_k = _cstm->get_word_count_in_doc(word_id, doc_id);
			log_pw_old += _cstm->_compute_reduced_log_probability_document(word_id, doc_id, n_k, old_Zi, old_alpha_word);
			_old_alpha_words[doc_id] = old_alpha_word;
		}
		// 新しい単語ベクトルの尤度
		double g0 = _cstm->get_g0_of_word(word_id);
		double log_pw_new = 0;
		for(int doc_id = 0;doc_id < get_num_documents();doc_id++){
			double* doc_vec = _cstm->get_doc_vector(doc_id);
			double new_alpha_word = _cstm->_compute_alpha_word(new_word_vec, doc_vec, g0);
			double old_alpha_word = _old_alpha_words[doc_id];
			// Ziの計算を簡略化
			double old_Zi = _cstm->get_Zi(doc_id);
			double new_Zi = old_Zi - old_alpha_word + new_alpha_word;
			assert(old_Zi >= old_alpha_word);
			assert(new_Zi >= new_alpha_word);
			int n_k = _cstm->get_word_count_in_doc(word_id, doc_id);
			log_pw_new += _cstm->_compute_reduced_log_probability_document(word_id, doc_id, n_k, new_Zi, new_alpha_word);
			_Zi_cache[doc_id] = new_Zi;
		}
		assert(log_pw_old != 0);
		assert(log_pw_new != 0);
		// 事前分布
		double log_prior_old = _cstm->compute_log_prior_vector(old_word_vec);
		double log_prior_new = _cstm->compute_log_prior_vector(new_word_vec);
		assert(log_prior_old != 0);
		assert(log_prior_new != 0);
		// 採択率
		double log_acceptance_rate = log_pw_new + log_prior_new - log_pw_old - log_prior_old;
		double acceptance_ratio = std::min(1.0, fmath::expd(log_acceptance_rate));
		double bernoulli = Sampler::uniform(0, 1);
		if(bernoulli <= acceptance_ratio){
			_num_acceptance_word += 1;
			// 新しいベクトルをセット
			_cstm->set_word_vector(word_id, new_word_vec);
			for(int doc_id = 0;doc_id < get_num_documents();doc_id++){
				_cstm->set_Zi(doc_id, _Zi_cache[doc_id]);
			}
			return true;
		}
		_num_rejection_word += 1;
		return false;
	}
	void perform_mh_sampling_document(){
		compile_if_needed();
		// 更新する文書ベクトルをランダムに1つ選択
		_random_sampling_doc_index += 1;
		if(_random_sampling_doc_index == _random_doc_ids.size()){
			std::shuffle(_random_doc_ids.begin(), _random_doc_ids.end(), Sampler::mt);
			_random_sampling_doc_index = 0;
		}
		int doc_id = _random_doc_ids[_random_sampling_doc_index];
		double* old_vec = get_doc_vector(doc_id);
		double* new_vec = draw_doc_vector(old_vec);
		accept_document_vector_if_needed(new_vec, old_vec, doc_id);
		_num_doc_vec_sampled += 1;
		_num_updates_doc[doc_id] += 1;
	}
	bool accept_document_vector_if_needed(double* new_doc_vec, double* old_doc_vec, int doc_id){
		double original_Zi = _cstm->get_Zi(doc_id);
		// 以前の文書ベクトルの尤度
		double log_pw_old = _cstm->compute_log_probability_document(doc_id);
		// 新しい文書ベクトルの尤度
		_cstm->set_doc_vector(doc_id, new_doc_vec);
		_cstm->update_Zi(doc_id);
		double log_pw_new = _cstm->compute_log_probability_document(doc_id);
		assert(log_pw_old != 0);
		assert(log_pw_new != 0);
		// 事前分布
		double log_prior_old = _cstm->compute_log_prior_vector(old_doc_vec);
		double log_prior_new = _cstm->compute_log_prior_vector(new_doc_vec);
		// 採択率
		double log_acceptance_rate = log_pw_new + log_prior_new - log_pw_old - log_prior_old;
		double acceptance_ratio = std::min(1.0, fmath::expd(log_acceptance_rate));
		double bernoulli = Sampler::uniform(0, 1);
		if(bernoulli <= acceptance_ratio){
			_num_acceptance_doc += 1;
			return true;
		}
		// 元に戻す
		_cstm->set_doc_vector(doc_id, old_doc_vec);
		_cstm->set_Zi(doc_id, original_Zi);
		_num_rejection_doc += 1;
		return false;
	}
	void perform_mh_sampling_alpha0(){
		compile_if_needed();
		int doc_id = Sampler::uniform_int(0, _cstm->_num_documents - 1);
		double old_alpha0 = _cstm->get_alpha0();
		double new_alpha0 = _cstm->draw_alpha0(old_alpha0);
		accept_alpha0_if_needed(new_alpha0, old_alpha0);
	}
	bool accept_alpha0_if_needed(double new_alpha0, double old_alpha0){
		int num_docs = _dataset.size();
		// 以前のa0の尤度
		double log_pw_old= 0;
		for(int doc_id = 0;doc_id < num_docs;doc_id++){
			log_pw_old += _cstm->compute_log_probability_document(doc_id);
			_Zi_cache[doc_id] = _cstm->get_Zi(doc_id);
		}
		// 新しいa0の尤度
		_cstm->set_alpha0(new_alpha0);
		update_all_Zi();
		double log_pw_new = 0;
		for(int doc_id = 0;doc_id < num_docs;doc_id++){
			log_pw_new += _cstm->compute_log_probability_document(doc_id);
		}
		// 事前分布
		double log_prior_old = _cstm->compute_log_prior_alpha0(old_alpha0);
		double log_prior_new = _cstm->compute_log_prior_alpha0(new_alpha0);
		// 採択率
		double log_acceptance_rate = log_pw_new + log_prior_new - log_pw_old - log_prior_old;
		double acceptance_ratio = std::min(1.0, fmath::expd(log_acceptance_rate));
		double bernoulli = Sampler::uniform(0, 1);
		if(bernoulli <= acceptance_ratio){
			_num_acceptance_alpha0 += 1;
			return true;
		}
		_num_rejection_alpha0 += 1;
		// 元に戻す
		_cstm->set_alpha0(old_alpha0);
		for(int doc_id = 0;doc_id < num_docs;doc_id++){
			_cstm->set_Zi(doc_id, _Zi_cache[doc_id]);
		}
		return false;
	}
	bool load(string dirname){
		_vocab->load(dirname + "/cstm.vocab");
		if(_cstm->load(dirname + "/cstm.model") == false){
			return false;
		}
		_ndim_d = _cstm->_ndim_d;
		if(_old_vec_copy == NULL){
			_old_vec_copy = new double[_ndim_d];
		}
		if(_new_vec_copy == NULL){
			_new_vec_copy = new double[_ndim_d];
		}
		int num_docs = _cstm->_num_documents;
		if(_old_alpha_words == NULL){
			_old_alpha_words = new double[num_docs];
		}
		if(_Zi_cache == NULL){
			_Zi_cache = new double[num_docs];
		}
		return true;
	}
	void save(string dirname){
		_vocab->save(dirname + "/cstm.vocab");
		_cstm->save(dirname + "/cstm.model");
	}
	void debug_num_updates_word(){
		int max = 0;
		int min = -1;
		for(id word_id = 0;word_id < get_num_vocabulary();word_id++){
			int count = _num_updates_word[word_id];
			if(count > max){
				max = count;
			}
			if(min == -1 || count < min){
				min = count;
			}
		}
		cout << "max: " << max << ", min: " << min << endl;
	}
	void debug_num_updates_doc(){
		int max = 0;
		int min = -1;
		for(int doc_id = 0;doc_id < get_num_documents();doc_id++){
			int count = _num_updates_doc[doc_id];
			if(count > max){
				max = count;
			}
			if(min == -1 || count < min){
				min = count;
			}
		}
		cout << "max: " << max << ", min: " << min << endl;
	}
};

BOOST_PYTHON_MODULE(model){
	python::class_<PyCSTM>("cstm")
	.def(python::init<>())
	.def("add_document", &PyCSTM::add_document)
	.def("compile", &PyCSTM::compile)
	.def("reset_statistics", &PyCSTM::reset_statistics)
	.def("get_num_vocabulary", &PyCSTM::get_num_vocabulary)
	.def("get_num_documents", &PyCSTM::get_num_documents)
	.def("get_sum_word_frequency", &PyCSTM::get_sum_word_frequency)
	.def("get_ndim_d", &PyCSTM::get_ndim_d)
	.def("get_word_vectors", &PyCSTM::get_word_vectors)
	.def("get_doc_vectors", &PyCSTM::get_doc_vectors)
	.def("get_high_freq_words", &PyCSTM::get_high_freq_words)
	.def("get_mh_acceptance_rate_for_word_vector", &PyCSTM::get_mh_acceptance_rate_for_word_vector)
	.def("get_mh_acceptance_rate_for_doc_vector", &PyCSTM::get_mh_acceptance_rate_for_doc_vector)
	.def("get_mh_acceptance_rate_for_alpha0", &PyCSTM::get_mh_acceptance_rate_for_alpha0)
	.def("get_num_doc_vec_sampled", &PyCSTM::get_num_doc_vec_sampled)
	.def("get_num_word_vec_sampled", &PyCSTM::get_num_word_vec_sampled)
	.def("get_alpha0", &PyCSTM::get_alpha0)
	.def("set_ndim_d", &PyCSTM::set_ndim_d)
	.def("set_alpha0", &PyCSTM::set_alpha0)
	.def("set_sigma_u", &PyCSTM::set_sigma_u)
	.def("set_sigma_phi", &PyCSTM::set_sigma_phi)
	.def("set_sigma_alpha0", &PyCSTM::set_sigma_alpha0)
	.def("set_gamma_alpha_a", &PyCSTM::set_gamma_alpha_a)
	.def("set_gamma_alpha_b", &PyCSTM::set_gamma_alpha_b)
	.def("perform_mh_sampling_word", &PyCSTM::perform_mh_sampling_word)
	.def("perform_mh_sampling_document", &PyCSTM::perform_mh_sampling_document)
	.def("perform_mh_sampling_alpha0", &PyCSTM::perform_mh_sampling_alpha0)
	.def("compute_perplexity", &PyCSTM::compute_perplexity)
	.def("compute_log_likelihood_data", &PyCSTM::compute_log_likelihood_data)
	.def("debug_num_updates_word", &PyCSTM::debug_num_updates_word)
	.def("debug_num_updates_doc", &PyCSTM::debug_num_updates_doc)
	.def("load", &PyCSTM::load)
	.def("save", &PyCSTM::save);
}