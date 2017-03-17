#include <set>
#include "model.cpp"
using namespace std;

struct multiset_value_comparator {
	bool operator()(const pair<id, double> &a, const pair<id, double> &b) {
		return a.second > b.second;
	}   
};

void test_1(){
	PyCSTM* model = new PyCSTM();
	int doc_id;
	doc_id = model->add_document("./documents/doc1.txt");
	doc_id = model->add_document("./documents/doc2.txt");
	model->compile();

	dump_vec(model->_cstm->_doc_vectors[0], model->_cstm->_ndim_d);
	dump_vec(model->_cstm->_doc_vectors[1], model->_cstm->_ndim_d);
	for(int i = 0;i < 1000000;i++){
		// model->perform_mh_sampling_document();
		model->perform_mh_sampling_word();
		// double ppl_train = model->compute_perplexity_train();
		// double ppl_test = model->compute_perplexity_test();
		// cout << i << " PPL: " << ppl_train << endl;
		// model->_vocab->dump();

		if(i % 10000 == 0){
			for(const auto &elem: model->_cstm->_word_vectors){
				id word_id = elem.first;
				wstring word = model->_vocab->token_id_to_string(word_id);
				double* vec = elem.second;
				// wcout << word << endl;
				// dump_vec(vec, model->_cstm->_ndim_d);
			}
			cout << "Epoch " << i / 10000 << " PPL: " << model->compute_perplexity() << endl;
			cout << model->_num_acceptance / (double)(model->_num_acceptance + model->_num_rejection) << endl;
		}
	}
	dump_vec(model->_cstm->_doc_vectors[0], model->_cstm->_ndim_d);
	dump_vec(model->_cstm->_doc_vectors[1], model->_cstm->_ndim_d);
	cout << model->_num_acceptance / (double)(model->_num_acceptance + model->_num_rejection) << endl;
	model->_num_acceptance = 0;
	model->_num_rejection = 0;
	for(int doc_id = 0;doc_id < 2;doc_id++){
		double sum_alpha = 0;
		for(const auto &elem: model->_docs_containing_word){
			id word_id = elem.first;
			double alpha = model->_cstm->compute_alpha_word_given_doc(word_id, doc_id);
			sum_alpha += alpha;
		}
		cout << "doc: " << doc_id << endl;
		for(const auto &elem: model->_docs_containing_word){
			id word_id = elem.first;
			wstring word = model->_vocab->token_id_to_string(word_id);
			double alpha = model->_cstm->compute_alpha_word_given_doc(word_id, doc_id);
			wcout << word << ": " << alpha / sum_alpha << endl;
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

int main(int argc, char *argv[]){
	for(int i = 0;i < 1;i++)
		test_1();
}
