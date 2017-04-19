# coding: utf-8
import argparse, sys, os, pylab, argparse
import numpy as np
sys.path.append(os.path.split(os.getcwd())[0])
import model

def main(args):
	assert os.path.exists(args.model_filename)
	cstm = model.cstm(args.model_filename)
	ndim_d = cstm.get_ndim_d()

	# 単語情報を取得
	words = cstm.get_words()
	for meta in words:
		# 単語ID、単語、総出現回数、単語ベクトル、この単語が含まれる文書ID
		word_id, word, count, vector, doc_ids = meta
		vector = np.asarray(vector, dtype=np.float32)
		print word_id, word, count

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model-filename", type=str, default="../model/cstm.model")
	args = parser.parse_args()
	main(args)