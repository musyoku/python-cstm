#include <boost/python.hpp>
#include "python/document.h"
#include "python/dictionary.h"
#include "python/dictionary.h"

using namespace cstm;
using namespace cstm::python;
using boost::python::arg;
using boost::python::args;

BOOST_PYTHON_MODULE(cstm){
	boost::python::class_<Dictionary>("dictionary")
	.def(boost::python::init<std::string>())
	.def("save", &Dictionary::save)
	.def("load", &Dictionary::load);
}