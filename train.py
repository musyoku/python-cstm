# coding: utf-8
import argparse, sys, os, re, time
import model
import numpy as np
from args import args
import dataset

class stdout:
	BOLD = "\033[1m"
	END = "\033[0m"
	CLEAR = "\033[2K"

def mkdir(target):
	assert target is not None
	try:
		os.mkdir(target)
	except:
		pass

def main():
	model_dir = "/".join(args.model_filename.split("/")[:-1])
	mkdir(model_dir)
	assert os.path.exists(args.document_dir)
	trainer = model.trainer()
	trainer.set_ndim_d(args.ndim_d)
	trainer.set_sigma_u(0.02)
	trainer.set_sigma_phi(0.04)
	trainer.set_sigma_alpha0(0.2)
	trainer.set_gamma_alpha_a(5)
	trainer.set_gamma_alpha_b(500)
	trainer.set_num_threads(args.num_thread)			# 文書ベクトルの更新に使うスレッド数
	trainer.set_ignore_word_count(args.ignore_count)	# 低頻度後を学習しない

	# 読み込み
	dataset.add_documents(trainer)
	trainer.compile()

	vocab_size = trainer.get_vocabulary_size()
	ignored_vocab_size = trainer.get_ignored_vocabulary_size()
	num_docs = trainer.get_num_documents()
	print "{} vocabularies ({} ignored), {} docs, {} words".format(vocab_size, ignored_vocab_size, num_docs, trainer.get_sum_word_frequency())
	start_time = time.time()
	total_time = 0
	itr = 0
	epoch = 0
	while True:
		# メトロポリス・ヘイスティングス法で単語ベクトル・文書ベクトルを更新
		# 内部的にはランダムに選択した文書・単語のベクトルを更新している
		trainer.perform_mh_sampling_document();
		trainer.perform_mh_sampling_word();
		if itr % 1000 == 0:
			trainer.perform_mh_sampling_alpha0()	# alpha0の更新は重いのでなるべくしたくない

		itr += 1
		if itr % 100 == 0:
			sys.stdout.write("\riteration {} / {}".format(itr, 10000))
			sys.stdout.flush()
		if itr % 10000 == 0:
			elapsed_time = time.time() - start_time
			print stdout.CLEAR + "\rEpoch", epoch
			print "	", trainer.compute_perplexity(), "ppl -", trainer.compute_log_likelihood_data(), "log likelihood -",  int((trainer.get_num_word_vec_sampled() + trainer.get_num_doc_vec_sampled())/ elapsed_time), "updates/sec", "-", int(elapsed_time), "sec", "-", int(total_time / 60.0), "min total"
			# 実際に受理・棄却された回数の統計を取ってあるので表示
			print "	MH acceptance:"
			print "		document:", trainer.get_mh_acceptance_rate_for_doc_vector(), ", word:", trainer.get_mh_acceptance_rate_for_word_vector(), ", a0:", trainer.get_mh_acceptance_rate_for_alpha0()
			print "	alpha0:", trainer.get_alpha0()

			# trainer._debug_num_updates_word()
			# trainer._debug_num_updates_doc()
			
			trainer.save(args.model_filename)
			trainer.reset_statistics()	# 統計をリセット. 結果表示用の統計なので学習とは無関係
			total_time += elapsed_time
			start_time = time.time()
			itr = 0
			epoch += 1

if __name__ == "__main__":
	main()