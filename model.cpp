#include <boost/python.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <string>
#include <unordered_map> 
#include "core/c_printf.h"
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
	}
};

BOOST_PYTHON_MODULE(model){
	python::class_<PyCSTM>("cstm")
	.def(python::init<>());
}