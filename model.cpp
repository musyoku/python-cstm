#include <boost/python.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <string>
#include <set>
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
	unordered_map<int, vector<vector<id>>> _dataset_train;
	unordered_map<int, vector<vector<id>>> _dataset_test;
	id* _word_ids;		// サンプリング用
	bool _is_ready;
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
	}
	~PyCSTM(){
		delete _cstm;
		delete _vocab;
		delete[] _word_ids;
	}
	void compile(){
		_cstm->compile();
		int num_words = _cstm->_word_vectors.size();
		// 単語IDのランダムサンプリング用テーブル
		_word_ids = new id[num_words];
		int index = 0;
		for(const auto &elem: _cstm->_word_vectors){
			id word_id = elem.first;
			_word_ids[index] = word_id;
			index += 1;
		}
		_is_ready = true;
	}
	int add_document(string filename, int train_split){
		wifstream ifs(filename.c_str());
		assert(ifs.fail() == false);
		int doc_id = _cstm->add_document();
		wstring sentence;
		vector<wstring> sentences;
		while (getline(ifs, sentence) && !sentence.empty()){
			assert(PyErr_CheckSignals() == 0);	// ctrl+cが押されたかチェック
			sentences.push_back(sentence);
		}
		assert(sentences.size() > train_split);
		vector<int> rand_indices;
		for(int i = 0;i < sentences.size();i++){
			rand_indices.push_back(i);
		}
		shuffle(rand_indices.begin(), rand_indices.end(), Sampler::mt);	// データをシャッフル
		for(int i = 0;i < rand_indices.size();i++){
			wstring &sentence = sentences[rand_indices[i]];
			if(i < train_split){
				add_train_sentence_to_doc(sentence, doc_id);
			}else{
				add_test_sentence_to_doc(sentence, doc_id);
			}
		}
		return doc_id;
	}
	void add_train_sentence_to_doc(wstring &sentence, int doc_id){
		auto itr = _dataset_train.find(doc_id);
		if(itr == _dataset_train.end()){
			vector<vector<id>> dataset;
			_dataset_train[doc_id] = dataset;
			itr = _dataset_train.find(doc_id);
		}
		vector<vector<id>> &dataset = itr->second;
		_add_sentence_to(sentence, dataset, doc_id);
	}
	void add_test_sentence_to_doc(wstring &sentence, int doc_id){
		auto itr = _dataset_test.find(doc_id);
		if(itr == _dataset_test.end()){
			vector<vector<id>> dataset;
			_dataset_test[doc_id] = dataset;
			itr = _dataset_test.find(doc_id);
		}
		vector<vector<id>> &dataset = itr->second;
		_add_sentence_to(sentence, dataset, doc_id);
	}
	void _add_sentence_to(wstring &sentence, vector<vector<id>> &dataset, int doc_id){
		vector<wstring> words;
		split_word_by(sentence, L' ', words);	// スペースで分割
		if(words.size() > 0){
			vector<id> tokens;
			for(auto word: words){
				if(word.size() == 0){
					continue;
				}
				id token_id = _vocab->add_string(word);
				tokens.push_back(token_id);
				_cstm->add_word(token_id, doc_id);
			}
			dataset.push_back(tokens);
		}
	}
	void perform_mh_sampling_word(){
		assert(_is_ready);
		int index = Sampler::uniform_int(0, _cstm->_word_vectors.size() - 1);
		id word_id = _word_ids[index];
		double* old_vec = _cstm->get_word_vec(word_id);
		double* new_vec = _cstm->draw_word_vec(old_vec);
		if(mh_accept_word_vec(new_vec, old_vec, word_id)){
			_cstm->set_word_vector(word_id, new_vec);
		}else{
			_cstm->set_word_vector(word_id, old_vec);
		}
	}
	bool mh_accept_word_vec(double* new_vec, double* old_vec, id word_id){
		set<int> &docs = _cstm->_docs_containing_word[word_id];
		assert(docs.size() > 0);
		double log_pw_old = 0;
		for(const int doc_id: docs){
			log_pw_old += _cstm->compute_Pw_given_doc(doc_id);
		}
		_cstm->set_word_vector(word_id, new_vec);
		double log_pw_new = 0;
		for(const int doc_id: docs){
			log_pw_new += _cstm->compute_Pw_given_doc(doc_id);
		}
		double log_t_given_old = _cstm->compute_log_Pvec_doc(new_vec, old_vec);
		double log_t_given_new = _cstm->compute_log_Pvec_doc(old_vec, new_vec);
		double log_adoption_rate = log_pw_new + log_t_given_new - log_pw_old - log_t_given_old;
		double adoption_rate = std::min(1.0, exp(log_adoption_rate));
		double bernoulli = Sampler::uniform(0, 1);
		if(bernoulli <= adoption_rate){
			return true;
		}
		return false;
	}
	void perform_mh_sampling_document(){
		assert(_is_ready);
		int doc_id = Sampler::uniform_int(0, _cstm->_num_documents - 1);
		double* old_vec = _cstm->get_doc_vec(doc_id);
		double* new_vec = _cstm->draw_doc_vec(old_vec);
		if(mh_accept_doc_vec(new_vec, old_vec, doc_id)){
			_cstm->set_doc_vector(doc_id, new_vec);
		}else{
			_cstm->set_doc_vector(doc_id, old_vec);
		}
	}
	bool mh_accept_doc_vec(double* new_vec, double* old_vec, int doc_id){
		double log_pw_old = _cstm->compute_Pw_given_doc(doc_id);
		_cstm->set_doc_vector(doc_id, new_vec);
		double log_pw_new = _cstm->compute_Pw_given_doc(doc_id);
		double log_t_given_old = _cstm->compute_log_Pvec_doc(new_vec, old_vec);
		double log_t_given_new = _cstm->compute_log_Pvec_doc(old_vec, new_vec);
		double log_adoption_rate = log_pw_new + log_t_given_new - log_pw_old - log_t_given_old;
		double adoption_rate = std::min(1.0, exp(log_adoption_rate));
		double bernoulli = Sampler::uniform(0, 1);
		if(bernoulli <= adoption_rate){
			return true;
		}
		return false;
	}
};

BOOST_PYTHON_MODULE(model){
	python::class_<PyCSTM>("cstm")
	.def(python::init<>());
}