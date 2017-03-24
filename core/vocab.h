#pragma once
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <string>
#include <fstream>
#include "common.h"
using namespace std;

class Vocab{
private:
	unordered_map<id, wstring> _string_by_token_id;
	unordered_map<id, id> _hash_to_id;
	hash<wstring> _hash_func;
	
public:
	Vocab(){
		
	}
	id add_string(wstring &str){
		id hash = hash_string(str);
		auto itr = _hash_to_id.find(hash);
		if(itr == _hash_to_id.end()){
			id token_id = _hash_to_id.size();
			_string_by_token_id[token_id] = str;
			_hash_to_id[hash] = token_id;
			return token_id;
		}
		return itr->second;
	}
	id hash_string(wstring &str){
		return (id)_hash_func(str);
	}
	wstring token_id_to_string(id token_id){
		auto itr = _string_by_token_id.find(token_id);
		assert(itr != _string_by_token_id.end());
		return itr->second;
	}
	wstring token_ids_to_sentence(vector<id> &token_ids){
		wstring sentence = L"";
		for(const auto &token_id: token_ids){
			wstring word = token_id_to_string(token_id);
			sentence += word;
			sentence += L" ";
		}
		return sentence;
	}
	int num_tokens(){
		return _string_by_token_id.size();
	}
	template <class Archive>
	void serialize(Archive& archive, unsigned int version)
	{
		archive & _string_by_token_id;
		archive & _hash_to_id;
	}
	void save(string filename){
		std::ofstream ofs(filename);
		boost::archive::binary_oarchive oarchive(ofs);
		oarchive << *this;
	}
	void load(string filename){
		std::ifstream ifs(filename);
		if(ifs.good()){
			boost::archive::binary_iarchive iarchive(ifs);
			iarchive >> *this;
		}
	}
	void dump(){
		for(auto elem : _string_by_token_id) {
			wcout << elem.first << ": " << elem.second << endl;
		} 
	}
};