# coding: utf-8
import argparse, sys, os, re, time
import model
import numpy as np

def mkdir(dir):
	assert dir is not None
	try:
		os.mkdir(dir)
	except:
		pass

def main(args):
	mkdir(args.model_dir)
	assert os.path.exists(args.model_dir)
	assert os.path.exists(args.document_dir)
	cstm = model.cstm()
	cstm.set_ndim_d(args.ndim_d)

	# 読み込み
	filelist = os.listdir(args.document_dir)
	filelist.sort()
	for filename in filelist:
		if re.search(r".txt$", filename):
			print "Loading {}".format(filename)
			cstm.add_document("{}/{}".format(args.document_dir, filename));
	cstm.compile()

	num_vocab = cstm.get_num_vocabulary()
	num_docs = cstm.get_num_documents()
	print "\r", num_vocab, "vocabularies,", num_docs, "docs,", cstm.get_sum_word_frequency(), "words"
	start_time = time.time()
	total_time = 0
	itr = 0
	epoch = 0
	while True:
		# MH法
		cstm.perform_mh_sampling_document();
		cstm.perform_mh_sampling_word();
		# if itr % 100 == 0:
		# 	cstm.perform_mh_sampling_alpha0()	# alpha0は頻繁に更新しない

		itr += 1
		if itr % 100 == 0:
			sys.stdout.write("\r{}/{}".format(itr, 10000))
			sys.stdout.flush()
		if itr % 10000 == 0:
			elapsed_time = time.time() - start_time
			print "\rEpoch", epoch, " " * 20
			print "	PPL:", int(cstm.compute_perplexity()), "-", int((cstm.get_num_word_vec_sampled() + cstm.get_num_doc_vec_sampled())/ elapsed_time), "updates/sec", "-", int(elapsed_time), "sec", "-", int(total_time / 60.0), "min"
			print "	MH acceptance:"
			print "		document:", cstm.get_mh_acceptance_rate_for_doc_vector(), ", word:", cstm.get_mh_acceptance_rate_for_word_vector()
			print "	alpha0:", cstm.get_alpha0()
			cstm.save(args.model_dir)
			cstm.reset_statistics()
			start_time = time.time()
			itr = 0
			epoch += 1

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model-dir", type=str, default="out")
	parser.add_argument("-t", "--document-dir", type=str, default="documents")
	parser.add_argument("-d", "--ndim-d", type=int, default=20)
	main(parser.parse_args())