#include <set>
#include <chrono>
#include <unordered_map>
#include "core/hashmap.h"
#include "core/sampler.h"
#include "model.cpp"
using namespace std;
using namespace emilib;

struct multiset_value_comparator {
	bool operator()(const pair<id, double> &a, const pair<id, double> &b) {
		return a.second > b.second;
	}   
};

void test_1(){
	PyCSTM* model = new PyCSTM();
	int doc_id;
	doc_id = model->add_document("./documents/0.txt");
	doc_id = model->add_document("./documents/1.txt");
	doc_id = model->add_document("./documents/2.txt");
	doc_id = model->add_document("./documents/3.txt");
	doc_id = model->add_document("./documents/4.txt");
	doc_id = model->add_document("./documents/5.txt");
	model->compile();

	int num_docs = model->get_num_documents();
	int num_words = model->get_num_vocabulary();
	int word_doc_ratio = (int)(num_words / (double)num_docs);

	for(int i = 1;i < 100000;i++){
		model->perform_mh_sampling_document();
		model->perform_mh_sampling_word();
		// model->perform_mh_sampling_alpha0();
		// if(i % 100 == 0){
		// }
		// double ppl_train = model->compute_perplexity_train();
		// double ppl_test = model->compute_perplexity_test();
		// cout << i << " PPL: " << ppl_train << endl;
		// model->_vocab->dump();

		if(i % 500 == 0){
			for(id word_id = 0;word_id < model->get_num_vocabulary();word_id++){
				wstring word = model->_vocab->token_id_to_string(word_id);
				double* vec = model->get_word_vector(word_id);
				// wcout << word << endl;
				// dump_vec(vec, model->_cstm->_ndim_d);
			}
			cout << "Epoch " << i / 500 << " PPL: " << model->compute_perplexity() << endl;
			cout << model->_num_acceptance_doc / (double)(model->_num_acceptance_doc + model->_num_rejection_doc) << ", ";
			model->_num_acceptance_doc = 0;
			model->_num_rejection_doc = 0;
			cout << model->_num_acceptance_word / (double)(model->_num_acceptance_word + model->_num_rejection_word) << ", ";
			model->_num_acceptance_word = 0;
			model->_num_rejection_word = 0;
			cout << model->_num_acceptance_alpha0 / (double)(model->_num_acceptance_alpha0 + model->_num_rejection_alpha0) << endl;
			model->_num_acceptance_alpha0 = 0;
			model->_num_rejection_alpha0 = 0;
			cout << "alpha0 <- " << model->_cstm->_alpha0 << endl;
		}
	}
	cout << model->_num_acceptance_doc / (double)(model->_num_acceptance_doc + model->_num_rejection_doc) << endl;
	for(int doc_id = 0;doc_id < model->get_num_documents();doc_id++){
		double sum_alpha = 0;
		for(const auto &elem: model->_docs_containing_word){
			id word_id = elem.first;
			double alpha = model->_cstm->compute_alpha_word_given_doc(word_id, doc_id);
			double n_k = model->_cstm->_n_k[doc_id][word_id];
			sum_alpha += alpha + n_k;
		}
		cout << "doc: " << doc_id << endl;
		for(const auto &elem: model->_docs_containing_word){
			id word_id = elem.first;
			wstring word = model->_vocab->token_id_to_string(word_id);
			double alpha = model->_cstm->compute_alpha_word_given_doc(word_id, doc_id);
			wcout << word << ": " << word_id << ": " << alpha / sum_alpha << endl;
		}
	}
	std::pair<id, double> pair;
	multiset<std::pair<id, double>, multiset_value_comparator> ranking;
	for(const auto &elem: model->_docs_containing_word){
		id word_id = elem.first;
		double* vec = model->get_word_vector(word_id);
		double distance = 0;
		for(int i = 0;i < model->_cstm->_ndim_d;i++){
			distance += vec[i] * vec[i];
		}
		distance = sqrt(distance);
		pair.first = word_id;
		pair.second = distance;
		ranking.insert(pair);
	}
	cout << "------ distance ------" << endl;
	for(const auto &elem: ranking){
		id word_id = elem.first;
		double distance = elem.second;
		wstring word = model->_vocab->token_id_to_string(word_id);
		wcout << word << ": " << distance << endl;
	}
	delete model;
}

void test2(){
	unordered_map<id, int> umap;
	HashMap<id, int> hmap;
    auto start = std::chrono::system_clock::now();
	for(int i = 0;i < 1000;i++){
		for(id c = 0;c < 100000;c++){
			umap[c] += 1;
		}
	}
    auto end = std::chrono::system_clock::now();
    auto diff = end - start;
    cout << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << endl;
    start = std::chrono::system_clock::now();
	for(int i = 0;i < 10;i++){
		for(id c = 0;c < 100000;c++){
			hmap[c] += 1;
		}
	}
    end = std::chrono::system_clock::now();
    diff = end - start;
    cout << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << endl;
}

void test3(){
	PyCSTM* model = new PyCSTM();
	model->add_document("./documents/doc1.txt");
	model->add_document("./documents/doc2.txt");
	model->compile();
	{
		double* vec = model->get_word_vector(10);
	    auto start = std::chrono::system_clock::now();
		uniform_real_distribution<double> distribution(0, 0.01);
		for(int i = 0;i < 1178528;i++){
			vec = model->draw_word_vector(vec);
		}
	    auto end = std::chrono::system_clock::now();
	    auto diff = end - start;
	    cout << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << endl;
	}
	{
		double* vec = model->get_word_vector(10);
	    auto start = std::chrono::system_clock::now();
		uniform_real_distribution<double> distribution(0, 0.01);
		for(int i = 0;i < 1178528;i++){
			vec = model->_cstm->draw_word_vector(vec);
		}
	    auto end = std::chrono::system_clock::now();
	    auto diff = end - start;
	    cout << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << endl;
	}
}

void test4(){
	for(int i = 0;i < 100;i++){
		cout << Sampler::uniform(0, 1) << endl;
	}
	exit(0);
}

void test5(){
	double alpha = 2.1;
	double n = 10;
	{
	    auto start = std::chrono::system_clock::now();
	    double a = 0;
		for(int i = 0;i < 20000000;i++){
			alpha += i * 0.0000001;
			a += lgamma(alpha + n) - lgamma(alpha);
		}
	    auto end = std::chrono::system_clock::now();
	    auto diff = end - start;
	    cout << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << endl;
		printf("%.16e\n", a);
	}
	alpha = 2.1;
	{
	    auto start = std::chrono::system_clock::now();
	    double a = 0;
		for(int i = 0;i < 20000000;i++){
			alpha += i * 0.0000001;
			double b = 0;
			for(int j = 0;j < n;j++){
				b += log(alpha + j);
			}
			a += b;
		}
	    auto end = std::chrono::system_clock::now();
	    auto diff = end - start;
	    cout << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << endl;
		printf("%.16e\n", a);
	}
	exit(0);
}

void test6(){
	PyCSTM* model = new PyCSTM();
	int doc_id;
	doc_id = model->add_document("./documents/doc1.txt");
	doc_id = model->add_document("./documents/doc2.txt");
	model->compile();
	for(int i = 1;i < 10000;i++){
		model->perform_mh_sampling_document();
		for(int j = 0;j < 10;j++){
			model->perform_mh_sampling_word();
		}
	}
	unordered_set<id> &word_set = model->_word_set[0];
	double log_pw = model->_cstm->compute_log_Pdocument(word_set, 0) / (double)(word_set.size());
	cout << word_set.size() << endl;
	cout << model->_sum_word_frequency[0] << endl;
	cout << exp(-log_pw) << endl;
	delete model;
	exit(0);
}

void test7(){
	double sum = 0;
	for(int i = 0;i < 10;i++){
	    auto start = std::chrono::system_clock::now();
		test6();
	    auto end = std::chrono::system_clock::now();
	    auto diff = end - start;
	    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
	    cout << elapsed_time << endl;
	    sum += elapsed_time;
	}
	cout << "avg: " << sum / 10.0 << endl;
	exit(0);
}

int main(int argc, char *argv[]){
	// test5();
	// test6();
	// test3();
    auto start = std::chrono::system_clock::now();
	for(int i = 0;i < 1;i++){
		test_1();
	}
    auto end = std::chrono::system_clock::now();
    auto diff = end - start;
    cout << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << endl;
}
