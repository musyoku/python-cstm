#include "model.cpp"
using namespace std;

void test_1(){
	PyCSTM* model = new PyCSTM();
	model->add_document("./documents/doc1.txt", 1);
	model->add_document("./documents/doc2.txt", 1);
	model->add_document("./documents/doc3.txt", 1);
	model->perform_mh_sampling_document();
	delete model;
}

int main(int argc, char *argv[]){
	for(int i = 0;i < 10;i++)
		test_1();
}
