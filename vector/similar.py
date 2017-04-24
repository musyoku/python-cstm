# coding: utf-8
import argparse, sys, os, pylab, argparse
import numpy as np
sys.path.append(os.path.split(os.getcwd())[0])
import model

def find_similar_words(args):
	assert os.path.exists(args.model_filename)
	cstm = model.cstm(args.model_filename)
	ndim_d = cstm.get_ndim_d()

	# 単語情報を取得
	words = cstm.get_words_similar_to_word(u"apple", 20)
	print "word_id	word		count	cosine"
	for meta in words:
		# 単語ID、単語、総出現回数、単語ベクトル、内積
		word_id, word, count, vector, cosine = meta
		vector = np.asarray(vector, dtype=np.float32)
		word = word.encode(sys.stdout.encoding) + " " * max(0, 8 - len(word))
		print "{}	{}	{}	{}".format(word_id, word, count, cosine)

def get_analogies(args):
	assert os.path.exists(args.model_filename)
	cstm = model.cstm(args.model_filename)
	ndim_d = cstm.get_ndim_d()

	king = np.asarray(cstm.get_word_vector_by_word(u"サーバルちゃん"), dtype=np.float32)
	man = np.asarray(cstm.get_word_vector_by_word(u"けもフレ"), dtype=np.float32)
	woman = np.asarray(cstm.get_word_vector_by_word(u"ごちうさ"), dtype=np.float32)
	queen = (king - man + woman).tolist()

	# 単語情報を取得
	words = cstm.get_words_similar_to_vector(queen, 20)
	print "word_id	word		count	cosine"
	for meta in words:
		# 単語ID、単語、総出現回数、単語ベクトル、内積
		word_id, word, count, vector, cosine = meta
		vector = np.asarray(vector, dtype=np.float32)
		word = word.encode(sys.stdout.encoding) + " " * max(0, 8 - len(word))
		print "{}	{}	{}	{}".format(word_id, word, count, cosine)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model-filename", type=str, default="../model/cstm.model")
	args = parser.parse_args()
	find_similar_words(args)
	get_analogies(args)