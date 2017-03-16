#include <boost/python.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <string>
#include <unordered_map> 
#include "core/c_printf.h"
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
	vector<vector<vector<id>>> _dataset_train;
	vector<vector<vector<id>>> _dataset_test;
	PyCSTM(){
		setlocale(LC_CTYPE, "ja_JP.UTF-8");
		ios_base::sync_with_stdio(false);
		locale default_loc("ja_JP.UTF-8");
		locale::global(default_loc);
		locale ctype_default(locale::classic(), default_loc, locale::ctype); //※
		wcout.imbue(ctype_default);
		wcin.imbue(ctype_default);
	}
	~PyCSTM(){
		delete _cstm;
	}
	int add_document(string filename, int train_split){
		wifstream ifs(filename.c_str());
		wstring sentence;
		if (ifs.fail()){
			exit(1);
		}
		vector<wstring> sentences;
		while (getline(ifs, sentence) && !sentence.empty()){
			if (PyErr_CheckSignals() != 0) {		// ctrl+cが押されたかチェック
				return;
			}
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
				add_train_data(sentence, doc_id);
			}else{
				add_test_data(sentence, doc_id);
			}
		}
	}
	void perform_mh_sampling_word(){
	}
	bool mh_accept_word(double* vec){
		
	}
	void perform_mh_sampling_document(){

	}
};

BOOST_PYTHON_MODULE(model){
	python::class_<PyCSTM>("cstm")
	.def(python::init<>());
}