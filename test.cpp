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
	for(int i = 0;i < 10000;i++){
		model->perform_mh_sampling_document();
		model->perform_mh_sampling_word();
	}
	dump_vec(model->_cstm->_doc_vectors[0], model->_cstm->_ndim_d);
	dump_vec(model->_cstm->_doc_vectors[1], model->_cstm->_ndim_d);
	dump_vec(model->_cstm->_doc_vectors[2], model->_cstm->_ndim_d);
	delete model;
}

int main(int argc, char *argv[]){
	for(int i = 0;i < 10;i++)
		test_1();
}
