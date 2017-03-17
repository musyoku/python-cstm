#include <boost/python.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <string>
#include <unordered_set>
#include <unordered_map> 
#include "core/cstm.h"
#include "core/vocab.h"
using namespace boost;

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
	vector<int> _sum_word_frequency;	// 文書ごとの単語の出現頻度の総和
	unordered_map<id, unordered_set<int>> _docs_containing_word;	// ある単語を含んでいる文書nのリスト
	id* _word_ids;			// サンプリング用
	bool _is_ready;
	double* _old_vec_copy;
	double* _new_vec_copy;
	double* _old_alpha_words;
	double* _original_Zi;
	int _num_acceptance;	// MH法で採択された回数
	int _num_rejection;		// MH法で棄却された回数
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
		_is_ready = false;
		_old_vec_copy = new double[_cstm->_ndim_d];
		_new_vec_copy = new double[_cstm->_ndim_d];
		_num_acceptance = 0;
		_num_rejection = 0;
	}
	~PyCSTM(){
		delete _cstm;
		delete _vocab;
		delete[] _word_ids;
		delete[] _old_vec_copy;
		delete[] _new_vec_copy;
		delete[] _old_alpha_words;
		delete[] _original_Zi;
	}
	void compile(){
		_cstm->compile();
		// 単語IDのランダムサンプリング用テーブル
		int num_docs = _dataset.size();
		int num_words = _docs_containing_word.size();
		_word_ids = new id[num_words];
		int index = 0;
		for(const auto &elem: _docs_containing_word){
			id word_id = elem.first;
			_word_ids[index] = word_id;
			index += 1;
		}
		assert(_sum_word_frequency.size() == _dataset.size());
		for(int i = 0;i < _sum_word_frequency.size();i++){
			cout << _sum_word_frequency[i] << ", ";
		}
		cout << endl;
		_old_alpha_words = new double[num_docs];
		_original_Zi = new double[num_docs];
		_is_ready = true;
	}
	int add_document(string filename){
		wifstream ifs(filename.c_str());
		assert(ifs.fail() == false);
		// 文書の追加
		int doc_id = _cstm->add_document();
		vector<vector<id>> dataset;
		_dataset.push_back(dataset);
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
				_cstm->add_word(word_id, doc_id);
				unordered_set<int> &docs = _docs_containing_word[word_id];
				docs.insert(doc_id);
			}
			dataset.push_back(word_ids);
		}
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
	double compute_perplexity(){
		double log_pw = 0;
		int n = 0;
		for(int doc_id = 0;doc_id < _dataset.size();doc_id++){
			vector<vector<id>> &dataset = _dataset[doc_id];
			log_pw += _cstm->compute_log_Pdocument(dataset, doc_id) / (double)_sum_word_frequency[doc_id];
		}
		return exp(-log_pw);
	}
	void perform_mh_sampling_word(){
		assert(_is_ready);
		int index = Sampler::uniform_int(0, _docs_containing_word.size() - 1);
		id word_id = _word_ids[index];
		double* old_vec = get_word_vector(word_id);
		double* new_vec = draw_word_vector(old_vec);
		if(mh_accept_word_vec(new_vec, old_vec, word_id)){
			_cstm->set_word_vector(word_id, new_vec);
		}else{
			_cstm->set_word_vector(word_id, old_vec);	// 元に戻す
			auto itr = _docs_containing_word.find(word_id);
			assert(itr != _docs_containing_word.end());
			unordered_set<int> &docs = itr->second;
			int cache_index = 0;
			for(const int doc_id: docs){
				_cstm->set_Zi(doc_id, _original_Zi[cache_index]);	// 元に戻す
				cache_index++;
			}
		}
	}
	bool mh_accept_word_vec(double* new_vec, double* old_vec, id word_id){
		auto itr = _docs_containing_word.find(word_id);
		assert(itr != _docs_containing_word.end());
		unordered_set<int> &docs = itr->second;
		assert(docs.size() > 0);
		// _cstm->set_word_vector(word_id, old_vec);
		double log_pw_old = 0;
		int cache_index = 0;
		for(const int doc_id: docs){
			_old_alpha_words[cache_index] = _cstm->compute_alpha_word_given_doc(word_id, doc_id);
			_original_Zi[cache_index] = _cstm->get_Zi(doc_id);
			vector<vector<id>> &dataset = _dataset[doc_id];
			log_pw_old += _cstm->compute_log_Pdocument(dataset, doc_id);
			cache_index++;
			// double alpha = _cstm->compute_alpha_word_given_doc(word_id, doc_id);
			// log_pw_old += log(alpha);
		}
		_cstm->set_word_vector(word_id, new_vec);	// 新しい単語ベクトルで差し替える
		double log_pw_new = 0;
		cache_index = 0;
		for(const int doc_id: docs){
			double old_alpha_word = _old_alpha_words[cache_index];
			double new_alpha_word = _cstm->compute_alpha_word_given_doc(word_id, doc_id);
			// cout << old_alpha_word << ", " << new_alpha_word << endl;
			_cstm->swap_Zi_component(doc_id, old_alpha_word, new_alpha_word);	// Ziの計算を簡略化
			vector<vector<id>> &dataset = _dataset[doc_id];
			log_pw_new += _cstm->compute_log_Pdocument(dataset, doc_id);
			// _cstm->swap_Zi_component(doc_id, new_alpha_word, old_alpha_word);	// 元に戻す
			cache_index++;
			// double alpha = _cstm->compute_alpha_word_given_doc(word_id, doc_id);
			// log_pw_old += log(alpha);
		}
		// double log_t_given_old = _cstm->compute_log_Pvector_doc(new_vec, old_vec);
		// double log_t_given_new = _cstm->compute_log_Pvector_doc(old_vec, new_vec);
		double log_prior_old = _cstm->compute_log_prior_vector(old_vec);
		double log_prior_new = _cstm->compute_log_prior_vector(new_vec);
		// dump_vec(old_vec, _cstm->_ndim_d);
		// dump_vec(new_vec, _cstm->_ndim_d);

		double log_acceptance_rate = log_pw_new + log_prior_new - log_pw_old - log_prior_old;
		// double log_acceptance_rate = log_prior_new - log_prior_old;
		// double log_acceptance_rate = log_pw_new - log_pw_old;
		double acceptance_ratio = std::min(1.0, exp(log_acceptance_rate));

		// if(acceptance_ratio < 1){
		// 	return false;
		// }
		// return true;
		double bernoulli = Sampler::uniform(0, 1);
		if(bernoulli <= acceptance_ratio){
			_num_acceptance += 1;
			return true;
		}
		_num_rejection += 1;
		return false;
	}
	void perform_mh_sampling_document(){
		assert(_is_ready);
		int doc_id = Sampler::uniform_int(0, _cstm->_num_documents - 1);
		double* old_vec = get_doc_vector(doc_id);
		double* new_vec = draw_doc_vector(old_vec);
		if(mh_accept_doc_vec(new_vec, old_vec, doc_id)){
			_cstm->set_doc_vector(doc_id, new_vec);
		}else{
			_cstm->set_doc_vector(doc_id, old_vec);
		}
	}
	bool mh_accept_doc_vec(double* new_vec, double* old_vec, int doc_id){
		vector<vector<id>> &dataset = _dataset[doc_id];
		_cstm->set_doc_vector(doc_id, old_vec);
		double log_pw_old = _cstm->compute_log_Pdocument(dataset, doc_id);
		_cstm->set_doc_vector(doc_id, new_vec);
		double log_pw_new = _cstm->compute_log_Pdocument(dataset, doc_id);
		double log_t_given_old = _cstm->compute_log_Pvector_doc(new_vec, old_vec);
		double log_t_given_new = _cstm->compute_log_Pvector_doc(old_vec, new_vec);
		double log_prior_old = _cstm->compute_log_prior_vector(old_vec);
		double log_prior_new = _cstm->compute_log_prior_vector(new_vec);
		
		double log_acceptance_rate = log_pw_new + log_prior_new - log_pw_old - log_prior_old;
		// double log_acceptance_rate = log_pw_new - log_pw_old;
		double acceptance_ratio = std::min(1.0, exp(log_acceptance_rate));
		double bernoulli = Sampler::uniform(0, 1);
		// if(acceptance_ratio < 1){
		// 	return false;
		// }
		// return true;
		if(bernoulli <= acceptance_ratio){
			_num_acceptance += 1;
			return true;
		}
		_num_rejection += 1;
		return false;
	}
};

BOOST_PYTHON_MODULE(model){
	python::class_<PyCSTM>("cstm")
	.def(python::init<>());
}