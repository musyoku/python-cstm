#include "model.cpp"
using namespace std;

void test_1(){
	PyCSTM* model = new PyCSTM();
	int doc_id = model->add_document("./documents/doc1.txt", 1);
	cout << doc_id << " added." << endl;
	doc_id = model->add_document("./documents/doc2.txt", 1);
	cout << doc_id << " added." << endl;
	doc_id = model->add_document("./documents/doc3.txt", 1);
	cout << doc_id << " added." << endl;
	model->compile();


	dump_vec(model->_cstm->_doc_vectors[0], model->_cstm->_ndim_d);
	dump_vec(model->_cstm->_doc_vectors[1], model->_cstm->_ndim_d);
	dump_vec(model->_cstm->_doc_vectors[2], model->_cstm->_ndim_d);
	for(int i = 0;i < 20000000;i++){
		model->perform_mh_sampling_document();
		model->perform_mh_sampling_word();

		if(i % 10000 == 0){
			for(const auto &elem: model->_cstm->_word_vectors){
				id word_id = elem.first;
				wstring word = model->_vocab->token_id_to_string(word_id);
				double* vec = elem.second;
				// wcout << word << endl;
				// dump_vec(vec, model->_cstm->_ndim_d);
			}
			cout << "Epoch " << i / 10000 << " PPL: " << model->compute_perplexity_train() << ", " << model->compute_perplexity_test() << endl;
		}
	}
	dump_vec(model->_cstm->_doc_vectors[0], model->_cstm->_ndim_d);
	dump_vec(model->_cstm->_doc_vectors[1], model->_cstm->_ndim_d);
	dump_vec(model->_cstm->_doc_vectors[2], model->_cstm->_ndim_d);

	for(const auto &elem: model->_cstm->_word_vectors){
		id word_id = elem.first;
		wstring word = model->_vocab->token_id_to_string(word_id);
		double* word_vec = elem.second;
		wcout << word << endl;
		dump_vec(word_vec, model->_cstm->_ndim_d);
		for(int doc_id = 0;doc_id < model->_dataset_train.size();doc_id++){
			double* doc_vec = model->_cstm->_doc_vectors[doc_id];
			double f = std::dot(word_vec, doc_vec, model->_cstm->_ndim_d);
			cout << "doc: " << doc_id << " f: " << f << endl;
		}
	}
	delete model;
}

int main(int argc, char *argv[]){
	for(int i = 0;i < 1;i++)
		test_1();
}
