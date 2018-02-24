CC = g++
BOOST = /usr/local/Cellar/boost/1.65.0
INCLUDE = `python3-config --includes` -std=c++14 -I$(BOOST)/include
LDFLAGS = `python3-config --ldflags` -lboost_serialization -lboost_python3 -L$(BOOST)/lib
SOFLAGS = -shared -fPIC -march=native
SOURCES = 	src/cstm/*.cpp

install: ## Python用ライブラリをビルドします.
	$(CC) $(INCLUDE) $(SOFLAGS) src/python/model.cpp $(SOURCES) $(LDFLAGS) -o run/cstm.so -O3 -DNDEBUG

test: ## LLDB用.
	$(CC) test.cpp $(INCLUDE) $(BOOST) $(FMATH) $(CFLAGS)

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.DEFAULT_GOAL := help