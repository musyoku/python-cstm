#pragma once
#include <string>
#include <unordered_map>
#include "../cstm/common.h"

namespace cstm {
	namespace python {
		class Dictionary{
		public:
			std::unordered_map<std::wstring, id> _map_word_to_id;	// すべての文字
			std::unordered_map<id, std::wstring> _map_id_to_word;	// すべての文字
			Dictionary();
			Dictionary(std::string filename);
			id add_word(std::wstring word);
			id get_word_id(std::wstring word);
			int get_num_words();
			bool load(std::string filename);
			bool save(std::string filename);
		};
	}
}