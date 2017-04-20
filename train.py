# coding: utf-8
import argparse, sys, os, re, time, argparse
import model
import numpy as np

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

def main(args):
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
	trainer.set_ignore_word_count(args.ignore_count)	# 低頻度後を学習しない場合

	# 読み込み
	filelist = os.listdir(args.document_dir)
	filelist.sort()
	for filename in filelist:
		if filename.endswith(".txt"):
			sys.stdout.write(stdout.CLEAR)
			sys.stdout.write("\rLoading {}".format(filename))
			sys.stdout.flush()
			doc_id = trainer.add_document("{}/{}".format(args.document_dir, filename));
	sys.stdout.write(stdout.CLEAR)
	sys.stdout.write("\r")

	# 全て追加し終わったら必ず呼ぶ. 必要なメモリが確保されて学習の準備ができる
	trainer.compile()

	vocab_size = trainer.get_vocabulary_size()
	ignored_vocab_size = trainer.get_ignored_vocabulary_size()
	num_docs = trainer.get_num_documents()
	print "語彙数:	{} (除外: {})".format(vocab_size, ignored_vocab_size)
	print "文書数:	{}".format(num_docs)
	print "単語数:	{}".format(trainer.get_sum_word_frequency())
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

		if itr % 10000 == 0:	# 10,000イテレーションごとに結果表示
			elapsed_time = time.time() - start_time
			print stdout.CLEAR
			print "\rEpoch", epoch
			print "	", trainer.compute_perplexity(), "ppl -", trainer.compute_log_likelihood_data(), "log likelihood -",  int((trainer.get_num_word_vec_sampled() + trainer.get_num_doc_vec_sampled())/ elapsed_time), "updates/sec", "-", int(elapsed_time), "sec", "-", int(total_time / 60.0), "min total"
			# 実際に受理・棄却された回数の統計を取ってあるので表示
			print "	MH acceptance:"
			print "		document:", trainer.get_mh_acceptance_rate_for_doc_vector(), ", word:", trainer.get_mh_acceptance_rate_for_word_vector(), ", a0:", trainer.get_mh_acceptance_rate_for_alpha0()
			print "	alpha0:", trainer.get_alpha0()

			# trainer._debug_num_updates_word()
			# trainer._debug_num_updates_doc()
			
			trainer.save(args.model_filename)
			trainer.reset_statistics()	# MH法などの統計をリセット. 結果表示用の統計なので学習とは無関係
			total_time += elapsed_time
			start_time = time.time()
			itr = 0
			epoch += 1

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model-filename", type=str, default="model/cstm.model")
	parser.add_argument("-d", "--document-dir", type=str, default="documents")
	parser.add_argument("-dim", "--ndim-d", type=int, default=20, help="ベクトルの次元数")
	parser.add_argument("-thread", "--num-thread", type=int, default=1)
	parser.add_argument("-ignore", "--ignore-count", type=int, default=0, help="これ以下の出現頻度の単語は学習しない")
	args = parser.parse_args()
	main(args)