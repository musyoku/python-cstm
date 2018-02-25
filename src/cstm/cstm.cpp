#include "cstm.h"

namespace cstm{
	CSTM::CSTM(int ndim_d, int vocabulary_size, int num_documents){
		_ndim_d = NDIM_D;
		_sigma_u = SIGMA_U;
		_sigma_phi = SIGMA_PHI;
		_sigma_alpha0 = SIGMA_ALPHA;
		_gamma_alpha_a = GAMMA_ALPHA_A;
		_gamma_alpha_b = GAMMA_ALPHA_B;
		_g0 = NULL;
		_word_vectors = NULL;
		_doc_vectors = NULL;
		_n_k = NULL;
		_word_count = NULL;
		_sum_n_k = NULL;
		_Zi = NULL;
		_log_likelihood_first_term = NULL;
		_tmp_vec = NULL;
		_vocabulary_size = 0;
		_num_documents = 0;
		_sum_word_frequency = 0;
		_ignore_word_count = 0;
		_standard_normal_distribution = std::normal_distribution<double>(0, 1);



		assert(_num_documents > 0);
		assert(_vocabulary_size > 0);
		assert(_sigma_u > 0);
		assert(_sigma_phi > 0);
		assert(_sigma_alpha0 > 0);
		// 基底分布
		for(id word_id = 0;word_id < _vocabulary_size;word_id++){
			double sum_count = 0;
			for(int doc_id = 0;doc_id < _num_documents;doc_id++){
				int* count = _n_k[doc_id];
				assert(count[word_id] >= 0);
				sum_count += count[word_id];
			}
			double g0 = sum_count / _sum_word_frequency;
			assert(0 < g0 && g0 <= 1);
			_g0[word_id] = g0;
			_word_count[word_id] = sum_count;
		}
		int sum_word_frequency_check = 0;
		for(int doc_id = 0;doc_id < _num_documents;doc_id++){
			int sum = 0;
			int* count = _n_k[doc_id];
			for(id word_id = 0;word_id < _vocabulary_size;word_id++){
				sum += count[word_id];
			}
			_sum_n_k[doc_id] = sum;
			sum_word_frequency_check += sum;
		}
		assert(sum_word_frequency_check == _sum_word_frequency);
		// 文書の単語集合の事後分布の式の最初の項はキャッシュできるのでしておく
		for(int doc_id = 0;doc_id < _num_documents;doc_id++){
			double log_pw = 0;
			for(int i = 2;i <= _sum_n_k[doc_id];i++){
				log_pw += log(i);
			}
			int* count = _n_k[doc_id];
			for(id word_id = 0;word_id < _vocabulary_size;word_id++){
				for(int i = 2;i <= count[word_id];i++){
					log_pw -= log(i);
				}
			}
			_log_likelihood_first_term[doc_id] = log_pw;
		}
		_noise_word = std::normal_distribution<double>(0, _sigma_phi);
		_noise_doc = std::normal_distribution<double>(0, _sigma_u);
		_noise_alpha0 = std::normal_distribution<double>(0, _sigma_alpha0);
	}
	CSTM::~CSTM(){
		_delete_capacity();
	}
	void CSTM::_allocate_capacity(int ndim_d, int vocabulary_size, int num_documents){
		assert(num_documents > 0);
		assert(vocabulary_size > 0);
		_ndim_d = ndim_d;
		_num_documents = num_documents;
		_vocabulary_size = vocabulary_size;
		// ハイパーパラメータ
		_alpha0 = 1;
		// メモリ確保
		_tmp_vec = generate_vector();
		_g0 = new double[vocabulary_size];
		_word_vectors = new double*[vocabulary_size];
		_doc_vectors = new double*[num_documents];
		_n_k = new int*[num_documents];
		_sum_n_k = new int[num_documents];
		_Zi = new double[num_documents];
		_log_likelihood_first_term = new double[num_documents];
		for(id word_id = 0;word_id < vocabulary_size;word_id++){
			_word_vectors[word_id] = generate_vector();
		}
		for(int doc_id = 0;doc_id < num_documents;doc_id++){
			_doc_vectors[doc_id] = generate_vector();
			_n_k[doc_id] = new int[vocabulary_size];
			_Zi[doc_id] = 0;
			for(id word_id = 0;word_id < vocabulary_size;word_id++){
				_n_k[doc_id][word_id] = 0;
			}
		}
		_word_count = new int[vocabulary_size];
	}
	void CSTM::_delete_capacity(){
		if(_word_vectors != NULL){
			for(id word_id = 0;word_id < _vocabulary_size;word_id++){
				if(_word_vectors[word_id] != NULL){
					delete[] _word_vectors[word_id];
				}
			}
		}
		if(_doc_vectors != NULL){
			for(int doc_id = 0;doc_id < _num_documents;doc_id++){
				if(_doc_vectors[doc_id] != NULL){
					delete[] _doc_vectors[doc_id];
				}
			}
		}
		if(_n_k != NULL){
			for(int doc_id = 0;doc_id < _num_documents;doc_id++){
				delete[] _n_k[doc_id];
			}
			delete[] _n_k;
		}
		if(_word_count != NULL){
			delete[] _word_count;
		}
		if(_tmp_vec != NULL){
			delete[] _tmp_vec;
		}
		if(_g0 != NULL){
			delete[] _g0;
		}
		if(_sum_n_k != NULL){
			delete[] _sum_n_k;
		}
		if(_Zi != NULL){
			delete[] _Zi;
		}
		if(_log_likelihood_first_term != NULL){
			delete[] _log_likelihood_first_term;
		}
	}
	void CSTM::add_word(id word_id, int doc_id){
		assert(doc_id < _num_documents);
		assert(word_id < _vocabulary_size);
		// カウントを更新
		int* count = _n_k[doc_id];
		count[word_id] += 1;
		_sum_word_frequency += 1;
	}
	double CSTM::generate_noise_from_standard_normal_distribution(){
		return _standard_normal_distribution(sampler::minstd);
	}
	double CSTM::generate_noise_doc(){
		return _noise_doc(sampler::minstd);
	}
	double CSTM::generate_noise_word(){
		return _noise_word(sampler::minstd);
	}
	double* CSTM::CSTM::generate_vector(){
		double* vec = new double[_ndim_d];
		for(int i = 0;i < _ndim_d;i++){
			vec[i] = generate_noise_from_standard_normal_distribution();
		}
		return vec;
	}
	double* CSTM::CSTM::draw_word_vector(double* old_vec){
		for(int i = 0;i < _ndim_d;i++){
			_tmp_vec[i] = old_vec[i] + generate_noise_word();
		}
		return _tmp_vec;
	}
	double* CSTM::CSTM::draw_doc_vector(double* old_vec){
		for(int i = 0;i < _ndim_d;i++){
			_tmp_vec[i] = old_vec[i] + generate_noise_doc();
		}
		return _tmp_vec;
	}
	double CSTM::draw_alpha0(double old_alpha0){
		double z = _noise_alpha0(sampler::minstd);
		return old_alpha0 * cstm::exp(z);
	}
	double CSTM::sum_alpha_word_given_doc(int doc_id){
		assert(doc_id < _num_documents);
		double sum = 0;
		for(id word_id = 0;word_id < _vocabulary_size;word_id++){
			sum += compute_alpha_word_given_doc(word_id, doc_id);
		}
		return sum;
	}
	double CSTM::compute_alpha_word_given_doc(id word_id, int doc_id){
		assert(word_id < _vocabulary_size);
		assert(doc_id < _num_documents);
		double* word_vec = _word_vectors[word_id];
		double* doc_vec = _doc_vectors[doc_id];
		double g0 = get_g0_of_word(word_id);
		return _compute_alpha_word(word_vec, doc_vec, g0);
	}
	double CSTM::_compute_alpha_word(double* word_vec, double* doc_vec, double g0){
		assert(word_vec != NULL);
		assert(doc_vec != NULL);
		assert(g0 > 0);
		double f = cstm::inner(word_vec, doc_vec, _ndim_d);
		double alpha = _alpha0 * g0 * cstm::exp(f);
		assert(alpha > 0);
		return alpha;
	}
	double CSTM::compute_reduced_log_probability_document(id word_id, int doc_id){
		assert(doc_id < _num_documents);
		assert(word_id < _vocabulary_size);
		double log_pw = 0;
		double Zi = _Zi[doc_id];
		int n_k = get_word_count_in_doc(word_id, doc_id);
		if(n_k == 0){
			return _compute_reduced_log_probability_document(word_id, doc_id, n_k, Zi, 0);
		}
		double alpha_k = compute_alpha_word_given_doc(word_id, doc_id);
		return _compute_reduced_log_probability_document(word_id, doc_id, n_k, Zi, alpha_k);
	}
	double CSTM::_compute_reduced_log_probability_document(id word_id, int doc_id, int n_k, double Zi, double alpha_k){
		assert(doc_id < _num_documents);
		assert(word_id < _vocabulary_size);
		double log_pw = 0;
		double sum_word_frequency = _sum_n_k[doc_id];
		log_pw += lgamma(Zi) - lgamma(Zi + sum_word_frequency);
		if(n_k == 0){
			return log_pw;
		}
		if(n_k > 10){
			// n_k > 10の場合はlgammaを使ったほうが速い
			log_pw += lgamma(alpha_k + n_k) - lgamma(alpha_k);
		}else{
			double tmp = 0;
			for(int i = 0;i < n_k;i++){
				tmp += log(alpha_k + i);
			}
			log_pw += tmp;
		}
		return log_pw;
	}
	double CSTM::compute_log_probability_document(int doc_id){
		assert(doc_id < _num_documents);
		double Zi = _Zi[doc_id];
		assert(Zi > 0);
		double sum_word_frequency = _sum_n_k[doc_id];
		double log_pw = lgamma(Zi) - lgamma(Zi + sum_word_frequency);
		for(id word_id = 0;word_id < _vocabulary_size;word_id++){
			int count = _word_count[word_id];
			if(count <= _ignore_word_count){
				continue;
			}
			log_pw += _compute_second_term_of_log_probability_document(doc_id, word_id);
		}
		return log_pw;
	}
	double CSTM::compute_log_probability_document_given_words(int doc_id, std::unordered_set<id> &word_ids){
		assert(doc_id < _num_documents);
		double Zi = 0;
		for(const id word_id: word_ids){
			Zi += compute_alpha_word_given_doc(word_id, doc_id);
		}
		assert(Zi > 0);
		double sum_word_frequency = _sum_n_k[doc_id];
		double log_pw = lgamma(Zi) - lgamma(Zi + sum_word_frequency);
		for(const id word_id: word_ids){
			int count = _word_count[word_id];
			if(count <= _ignore_word_count){
				continue;
			}
			log_pw += _compute_second_term_of_log_probability_document(doc_id, word_id);
		}
		return log_pw;
	}
	double CSTM::_compute_second_term_of_log_probability_document(int doc_id, id word_id){
		double alpha_k = compute_alpha_word_given_doc(word_id, doc_id);
		int n_k = get_word_count_in_doc(word_id, doc_id);
		if(n_k > 10){
			// n_k > 10の場合はlgammaを使ったほうが速い
			return lgamma(alpha_k + n_k) - lgamma(alpha_k);
		}
		double tmp = 0;
		for(int i = 0;i < n_k;i++){
			tmp += log(alpha_k + i);
		}
		return tmp;
	}
	double CSTM::compute_log_prior_alpha0(double alpha0){
		return _gamma_alpha_a * log(_gamma_alpha_b) - lgamma(_gamma_alpha_a) + (_gamma_alpha_a - 1) * log(alpha0) - _gamma_alpha_b * alpha0;
	}
	double CSTM::compute_log_Pvector_doc(double* new_vec, double* old_vec){
		return _compute_log_Pvector_given_sigma(new_vec, old_vec, _sigma_u);
	}
	double CSTM::compute_log_Pvector_word(double* new_vec, double* old_vec){
		return _compute_log_Pvector_given_sigma(new_vec, old_vec, _sigma_phi);
	}
	double CSTM::_compute_log_Pvector_given_sigma(double* new_vec, double* old_vec, double sigma){
		double log_pvec = (double)_ndim_d * log(1.0 / (sqrt(2.0 * PI) * sigma));
		for(int i = 0;i < _ndim_d;i++){
			log_pvec -= (new_vec[i] - old_vec[i]) * (new_vec[i] - old_vec[i]) / (2.0 * sigma * sigma);		
		}
		return log_pvec;
	}
	double CSTM::compute_log_prior_vector(double* vec){
		double log_pvec = (double)_ndim_d * log(1.0 / (sqrt(2.0 * PI)));
		for(int i = 0;i < _ndim_d;i++){
			log_pvec -= vec[i] * vec[i] * 0.5;
		}
		return log_pvec;
	}
	double CSTM::get_alpha0(){
		return _alpha0;
	}
	double CSTM::get_g0_of_word(id word_id){
		assert(word_id < _vocabulary_size);
		return _g0[word_id];
	}
	int CSTM::get_sum_word_frequency_of_doc(int doc_id){
		assert(doc_id < _num_documents);
		return _sum_n_k[doc_id];
	}
	double* CSTM::get_doc_vector(int doc_id){
		assert(doc_id < _num_documents);
		return _doc_vectors[doc_id];
	}
	double* CSTM::get_word_vector(id word_id){
		assert(word_id < _vocabulary_size);
		double* vec = _word_vectors[word_id];
		assert(vec != NULL);
		return vec;
	}
	int CSTM::get_word_count_in_doc(id word_id, int doc_id){
		assert(doc_id < _num_documents);
		assert(word_id < _vocabulary_size);
		int* count = _n_k[doc_id];
		return count[word_id];
	}
	int CSTM::get_word_count(id word_id){
		assert(word_id < _vocabulary_size);
		return _word_count[word_id];
	}
	int CSTM::get_ignore_word_count(){
		return _ignore_word_count;
	}
	double CSTM::get_Zi(int doc_id){
		assert(doc_id < _num_documents);
		return _Zi[doc_id];
	}
	void CSTM::set_ndim_d(int ndim_d){
		_ndim_d = ndim_d;
	}
	void CSTM::set_alpha0(double alpha0){
		_alpha0 = alpha0;
	}
	void CSTM::set_sigma_u(double sigma_u){
		_sigma_u = sigma_u;
	}
	void CSTM::set_sigma_phi(double sigma_phi){
		_sigma_phi = sigma_phi;
	}
	void CSTM::set_sigma_alpha(double sigma_alpha){
		_sigma_alpha0 = sigma_alpha;
	}
	void CSTM::set_gamma_alpha_a(double gamma_alpha_a){
		_gamma_alpha_a = gamma_alpha_a;
	}
	void CSTM::set_gamma_alpha_b(double gamma_alpha_b){
		_gamma_alpha_b = gamma_alpha_b;
	}
	void CSTM::set_num_documents(int num_documents){
		_num_documents = num_documents;
	}
	void CSTM::set_size_vocabulary(int vocabulary_size){
		_vocabulary_size = vocabulary_size;
	}
	void CSTM::set_ignore_word_count(int count){
		_ignore_word_count = count;
	}
	void CSTM::set_word_vector(id word_id, double* source){
		assert(word_id < _vocabulary_size);
		double* target = _word_vectors[word_id];
		assert(target != NULL);
		std::memcpy(target, source, _ndim_d * sizeof(double));
	}
	void CSTM::set_doc_vector(int doc_id, double* source){
		assert(doc_id < _num_documents);
		double* target = _doc_vectors[doc_id];
		std::memcpy(target, source, _ndim_d * sizeof(double));
	}
	void CSTM::update_Zi(int doc_id){
		set_Zi(doc_id, sum_alpha_word_given_doc(doc_id));
	}
	void CSTM::set_Zi(int doc_id, double new_value){
		assert(doc_id < _num_documents);
		assert(new_value > 0);
		_Zi[doc_id] = new_value;
	}
	// template <class Archive>
	// void CSTM::serialize(Archive &archive, unsigned int version)
	// {
	// 	boost::serialization::split_free(archive, *this, version);
	// }
	// void CSTM::save(std::string filename){
	// 	std::ofstream ofs(filename);
	// 	boost::archive::binary_oarchive oarchive(ofs);
	// 	oarchive << *this;
	// }
	// bool CSTM::load(std::string filename){
	// 	std::ifstream ifs(filename);
	// 	if(ifs.good()){
	// 		boost::archive::binary_iarchive iarchive(ifs);
	// 		iarchive >> *this;
	// 		return true;
	// 	}
	// 	return false;
	// }
	template <class Archive>
	void CSTM::serialize(Archive &archive, unsigned int version)
	{
		boost::serialization::split_member(archive, *this, version);
	}
	template void CSTM::serialize(boost::archive::binary_iarchive &ar, unsigned int version);
	template void CSTM::serialize(boost::archive::binary_oarchive &ar, unsigned int version);
	void CSTM::save(boost::archive::binary_oarchive &archive, unsigned int version) const {
		archive & _ndim_d;
		archive & _num_documents;
		archive & _vocabulary_size;
		archive & _sum_word_frequency;
		archive & _sigma_u;
		archive & _sigma_phi;
		archive & _sigma_alpha0;
		archive & _alpha0;
		for(int doc_id = 0;doc_id < _num_documents;doc_id++){
			archive & _Zi[doc_id];
			archive & _sum_n_k[doc_id];
			archive & _log_likelihood_first_term[doc_id];
		}
		for(int doc_id = 0;doc_id < _num_documents;doc_id++){
			for(id word_id = 0;word_id < _vocabulary_size;word_id++){
				archive & _n_k[doc_id][word_id];
			}
		}
		for(int doc_id = 0;doc_id < _num_documents;doc_id++){
			double* vec = _doc_vectors[doc_id];
			for(int i = 0;i < _ndim_d;i++){
				archive & vec[i];
			}
		}
		for(id word_id = 0;word_id < _vocabulary_size;word_id++){
			double* vec = _word_vectors[word_id];
			for(int i = 0;i < _ndim_d;i++){
				archive & vec[i];
			}
		}
		for(id word_id = 0;word_id < _vocabulary_size;word_id++){
			archive & _word_count[word_id];
		}
		for(id word_id = 0;word_id < _vocabulary_size;word_id++){
			archive & _g0[word_id];
		}
	}
	void CSTM::load(boost::archive::binary_iarchive &archive, unsigned int version) {
		archive & _ndim_d;
		archive & _num_documents;
		archive & _vocabulary_size;
		archive & _sum_word_frequency;
		archive & _sigma_u;
		archive & _sigma_phi;
		archive & _sigma_alpha0;
		archive & _alpha0;
		if(_word_vectors == NULL){
			_word_vectors = new double*[_vocabulary_size];
			for(id word_id = 0;word_id < _vocabulary_size;word_id++){
				_word_vectors[word_id] = new double[_ndim_d];
			}
		}
		if(_doc_vectors == NULL){
			_doc_vectors = new double*[_num_documents];
			for(int doc_id = 0;doc_id < _num_documents;doc_id++){
				_doc_vectors[doc_id] = new double[_ndim_d];
			}
		}
		if(_n_k == NULL){
			_n_k = new int*[_num_documents];
			for(int doc_id = 0;doc_id < _num_documents;doc_id++){
				_n_k[doc_id] = new int[_vocabulary_size];
			}
		}
		if(_sum_n_k == NULL){
			_sum_n_k = new int[_num_documents];
		}
		if(_Zi == NULL){
			_Zi = new double[_num_documents];
		}
		if(_log_likelihood_first_term == NULL){
			_log_likelihood_first_term = new double[_num_documents];
		}
		if(_word_count == NULL){
			_word_count = new int[_vocabulary_size];
		}

		for(int doc_id = 0;doc_id < _num_documents;doc_id++){
			archive & _Zi[doc_id];
			archive & _sum_n_k[doc_id];
			archive & _log_likelihood_first_term[doc_id];
		}
		for(int doc_id = 0;doc_id < _num_documents;doc_id++){
			for(id word_id = 0;word_id < _vocabulary_size;word_id++){
				archive & _n_k[doc_id][word_id];
			}
		}
		for(int doc_id = 0;doc_id < _num_documents;doc_id++){
			double* vec = _doc_vectors[doc_id];
			for(int i = 0;i < _ndim_d;i++){
				archive & vec[i];
			}
		}
		for(id word_id = 0;word_id < _vocabulary_size;word_id++){
			double* vec = _word_vectors[word_id];
			for(int i = 0;i < _ndim_d;i++){
				archive & vec[i];
			}
		}
		for(id word_id = 0;word_id < _vocabulary_size;word_id++){
			archive & _word_count[word_id];
		}
		if(_g0 == NULL){
			_g0 = new double[_vocabulary_size];
		}
		for(id word_id = 0;word_id < _vocabulary_size;word_id++){
			archive & _g0[word_id];
		}
	}
}
// モデルの保存用
// namespace boost { namespace serialization {
// template<class Archive>
// 	void save(Archive &archive, const cstm::CSTM &cstm, unsigned int version) {
// 		archive & cstm._ndim_d;
// 		archive & cstm._num_documents;
// 		archive & cstm._vocabulary_size;
// 		archive & cstm._sum_word_frequency;
// 		archive & cstm._sigma_u;
// 		archive & cstm._sigma_phi;
// 		archive & cstm._sigma_alpha0;
// 		archive & cstm._alpha0;
// 		archive & cstm._is_initialized;
// 		archive & cstm._is_compiled;
// 		if(cstm._is_initialized){
// 			for(int doc_id = 0;doc_id < cstm._num_documents;doc_id++){
// 				archive & cstm._Zi[doc_id];
// 				archive & cstm._sum_n_k[doc_id];
// 				archive & cstm._log_likelihood_first_term[doc_id];
// 			}
// 			for(int doc_id = 0;doc_id < cstm._num_documents;doc_id++){
// 				for(id word_id = 0;word_id < cstm._vocabulary_size;word_id++){
// 					archive & cstm._n_k[doc_id][word_id];
// 				}
// 			}
// 			for(int doc_id = 0;doc_id < cstm._num_documents;doc_id++){
// 				double* vec = cstm._doc_vectors[doc_id];
// 				for(int i = 0;i < cstm._ndim_d;i++){
// 					archive & vec[i];
// 				}
// 			}
// 			for(id word_id = 0;word_id < cstm._vocabulary_size;word_id++){
// 				double* vec = cstm._word_vectors[word_id];
// 				for(int i = 0;i < cstm._ndim_d;i++){
// 					archive & vec[i];
// 				}
// 			}
// 			for(id word_id = 0;word_id < cstm._vocabulary_size;word_id++){
// 				archive & cstm._word_count[word_id];
// 			}
// 		}
// 		if(cstm._is_compiled){
// 			for(id word_id = 0;word_id < cstm._vocabulary_size;word_id++){
// 				archive & cstm._g0[word_id];
// 			}
// 		}
// 	}
// template<class Archive>
// 	void load(Archive &archive, cstm::CSTM &cstm, unsigned int version) {
// 		archive & cstm._ndim_d;
// 		archive & cstm._num_documents;
// 		archive & cstm._vocabulary_size;
// 		archive & cstm._sum_word_frequency;
// 		archive & cstm._sigma_u;
// 		archive & cstm._sigma_phi;
// 		archive & cstm._sigma_alpha0;
// 		archive & cstm._alpha0;
// 		archive & cstm._is_initialized;
// 		archive & cstm._is_compiled;
// 		if(cstm._is_initialized){
// 			if(cstm._word_vectors == NULL){
// 				cstm._word_vectors = new double*[cstm._vocabulary_size];
// 				for(id word_id = 0;word_id < cstm._vocabulary_size;word_id++){
// 					cstm._word_vectors[word_id] = new double[cstm._ndim_d];
// 				}
// 			}
// 			if(cstm._doc_vectors == NULL){
// 				cstm._doc_vectors = new double*[cstm._num_documents];
// 				for(int doc_id = 0;doc_id < cstm._num_documents;doc_id++){
// 					cstm._doc_vectors[doc_id] = new double[cstm._ndim_d];
// 				}
// 			}
// 			if(cstm._n_k == NULL){
// 				cstm._n_k = new int*[cstm._num_documents];
// 				for(int doc_id = 0;doc_id < cstm._num_documents;doc_id++){
// 					cstm._n_k[doc_id] = new int[cstm._vocabulary_size];
// 				}
// 			}
// 			if(cstm._sum_n_k == NULL){
// 				cstm._sum_n_k = new int[cstm._num_documents];
// 			}
// 			if(cstm._Zi == NULL){
// 				cstm._Zi = new double[cstm._num_documents];
// 			}
// 			if(cstm._log_likelihood_first_term == NULL){
// 				cstm._log_likelihood_first_term = new double[cstm._num_documents];
// 			}
// 			if(cstm._word_count == NULL){
// 				cstm._word_count = new int[cstm._vocabulary_size];
// 			}

// 			for(int doc_id = 0;doc_id < cstm._num_documents;doc_id++){
// 				archive & cstm._Zi[doc_id];
// 				archive & cstm._sum_n_k[doc_id];
// 				archive & cstm._log_likelihood_first_term[doc_id];
// 			}
// 			for(int doc_id = 0;doc_id < cstm._num_documents;doc_id++){
// 				for(id word_id = 0;word_id < cstm._vocabulary_size;word_id++){
// 					archive & cstm._n_k[doc_id][word_id];
// 				}
// 			}
// 			for(int doc_id = 0;doc_id < cstm._num_documents;doc_id++){
// 				double* vec = cstm._doc_vectors[doc_id];
// 				for(int i = 0;i < cstm._ndim_d;i++){
// 					archive & vec[i];
// 				}
// 			}
// 			for(id word_id = 0;word_id < cstm._vocabulary_size;word_id++){
// 				double* vec = cstm._word_vectors[word_id];
// 				for(int i = 0;i < cstm._ndim_d;i++){
// 					archive & vec[i];
// 				}
// 			}
// 			for(id word_id = 0;word_id < cstm._vocabulary_size;word_id++){
// 				archive & cstm._word_count[word_id];
// 			}
// 		}
// 		if(cstm._is_compiled){
// 			if(cstm._g0 == NULL){
// 				cstm._g0 = new double[cstm._vocabulary_size];
// 			}
// 			for(id word_id = 0;word_id < cstm._vocabulary_size;word_id++){
// 				archive & cstm._g0[word_id];
// 			}
// 		}
// 	}
// }} // namespace boost::serialization