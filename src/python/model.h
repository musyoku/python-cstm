#include "../cstm/cstm.h"

namespace cstm {
	namespace python {
		class Model {
		public:
			cstm::CSTM* _cstm;
			Model(int ndim_d,
				  int vocabulary_size, 
				  int num_docs);
			Model(std::string filename);
			~Model();
			bool load(std::string filename);
			bool save(std::string filename);
		};
	}
}