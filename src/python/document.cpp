#include "document.h"

namespace cstm {
	namespace python {
		void Document::add_sentence(std::vector<std::wstring> &raw_sentence, Dictionary* dictionary){
			std::vector<id> sentence;
			for(auto word: raw_sentence){
				id word_id = dictionary->add_word(word);
				sentence.push_back(word_id);
			}
		}
	}
}