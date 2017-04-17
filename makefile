CC = icpc
INCLUDE = -I`python -c 'from distutils.sysconfig import *; print get_python_inc()'`
BOOST = -lboost_python -lpython2.7 -lboost_serialization
FMATH =  -fomit-frame-pointer -fno-operator-names -msse2 -mfpmath=sse -march=native
CFLAGS = -std=c++11 -L/usr/local/lib -O3
CFLAGS_SO = -shared -fPIC -std=c++11 -L/usr/local/lib -O3 

install: ## Python用ライブラリをビルドします.
	$(CC) model.cpp -o model.so $(INCLUDE) $(BOOST) $(FMATH) $(CFLAGS_SO)

test: ## LLDB用.
	$(CC) test.cpp $(INCLUDE) $(BOOST) $(FMATH) $(CFLAGS)

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.DEFAULT_GOAL := help