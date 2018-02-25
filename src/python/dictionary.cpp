#include <boost/serialization/serialization.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <fstream>
#include <iostream>
#include "dictionary.h"

namespace cstm {
	namespace python {
		Dictionary::Dictionary(){}
		Dictionary::Dictionary(std::string filename){
			if(load(filename) == false){
				std::cout << filename << " not found." << std::endl;
				exit(0);
			}
		}
		id Dictionary::add_word(std::wstring word){
			auto itr = _map_word_to_id.find(word);
			if(itr == _map_word_to_id.end()){
				id word_id = _map_word_to_id.size();
				_map_word_to_id[word] = word_id;
				_map_id_to_word[word_id] = word;
				return word_id;
			}
			return itr->second;
		}
		id Dictionary::get_word_id(std::wstring word){
			auto itr = _map_word_to_id.find(word);
			if(itr == _map_word_to_id.end()){
				return SPECIAL_WORD_UNK;
			}
			return itr->second;
		}
		int Dictionary::get_num_words(){
			return _map_word_to_id.size();
		}
		bool Dictionary::load(std::string filename){
			std::string dictionary_filename = filename;
			std::ifstream ifs(dictionary_filename);
			if(ifs.good()){
				boost::archive::binary_iarchive iarchive(ifs);
				iarchive >> _map_word_to_id;
				iarchive >> _map_id_to_word;
				ifs.close();
				return true;
			}
			ifs.close();
			return false;
		}
		bool Dictionary::save(std::string filename){
			std::ofstream ofs(filename);
			boost::archive::binary_oarchive oarchive(ofs);
			oarchive << _map_word_to_id;
			oarchive << _map_id_to_word;
			ofs.close();
			return true;
		}
	}
}