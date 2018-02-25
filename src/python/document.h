#include <vector>
#include "dictionary.h"

namespace cstm {
	namespace python {
		class Document {
		public:
			std::vector<std::vector<id>> _sentences;
			void add_sentence(std::vector<std::wstring> &raw_sentence, Dictionary* dictionary);
		};
	}
}