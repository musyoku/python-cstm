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
	for filename in os.listdir(args.document_dir):
		if re.search(r".txt$", filename):
			sys.stdout.write("\rLoading {}".format(filename))
			sys.stdout.flush()
			cstm.add_document("{}/{}".format(args.document_dir, filename));
	cstm.compile()

	print "\r", cstm.get_num_vocabulary(), "vocabularies,", cstm.get_num_documents(), "docs,", cstm.get_sum_word_frequency(), "words"
	start_time = time.time()
	itr = 0
	while True:
		# MH法
		cstm.perform_mh_sampling_document();
		cstm.perform_mh_sampling_word();
		itr += 1
		if itr % 10000 == 0:
			elapsed_time = time.time() - start_time
			print "PPL:", int(cstm.compute_perplexity()), "-", int((cstm.get_num_word_vec_sampled() + cstm.get_num_doc_vec_sampled())/ elapsed_time), "updates/sec", "-", int(elapsed_time), "sec"
			print "MH acceptance:"
			print "	document:", cstm.get_mh_acceptance_rate_for_doc_vector(), ", word:", cstm.get_mh_acceptance_rate_for_word_vector()
			cstm.save(args.model_dir)
			cstm.reset_statistics()
			start_time = time.time()
			itr = 0

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model-dir", type=str, default="out")
	parser.add_argument("-t", "--document-dir", type=str, default="documents")
	main(parser.parse_args())