# coding: utf-8
import argparse, sys, os, pylab
import seaborn as sns
import numpy as np
import model
import argparse

# フォントをセット
# UbuntuならTakaoGothicなどが標準で入っている
if sys.platform == "darwin":
	fontfamily = "MS Gothic"
else:
	fontfamily = "TakaoGothic"
sns.set(font=[fontfamily], font_scale=1)

def mkdir(target):
	assert target is not None
	try:
		os.mkdir(target)
	except:
		pass

def main():
	mkdir(args.output_dir)
	assert os.path.exists(args.model_dir)
	cstm = model.cstm()
	assert cstm.load(args.model_dir) == True
	ndim = cstm.get_ndim_d()
	# ベクトルを取得
	word_vectors = np.asarray(cstm.get_word_vectors(), dtype=np.float32)
	doc_vectors = np.asarray(cstm.get_doc_vectors(), dtype=np.float32)
	# print doc_vectors
	# print word_vectors

	# ベクトルの各次元の平均、標準偏差
	for i in xrange(ndim):
		print np.mean(word_vectors[:, i]), np.std(word_vectors[:, i])
	for i in xrange(ndim):
		print np.mean(doc_vectors[:, i]), np.std(doc_vectors[:, i])

	# fの平均、標準偏差
	for doc_id in xrange(cstm.get_num_documents()):
		doc_vector = doc_vectors[doc_id]
		f = np.inner(word_vectors, doc_vector)
		print np.mean(f), np.std(f)

	# 単語ベクトル・文書ベクトルのそれぞれの次元についてプロット
	for i in xrange(ndim - 1):
		plot_kde(word_vectors[:,i:], args.output_dir, filename="word_kde_{}-{}".format(i, i + 1))
		plot_scatter(word_vectors[:,i:], args.output_dir, filename="word_scatter_{}-{}".format(i, i + 1))
		plot_kde(doc_vectors[:,i:], args.output_dir, filename="doc_kde_{}-{}".format(i, i + 1))
		plot_scatter(doc_vectors[:,i:], args.output_dir, filename="doc_scatter_{}-{}".format(i, i + 1))

	# 文書にグループがある場合
	documents_for_group, group_for_document = dataset.load_groups()
	if len(documents_for_group) > 0:
		doc_vectors_for_category = []
		for i, (group_name, doc_ids) in enumerate(documents_for_group.items()):
			doc_vectors_for_category.append([])
			for doc_id in doc_ids:
				doc_vectors_for_category[i].append(doc_vectors[doc_id])
		doc_vectors_for_category = np.asanyarray(doc_vectors_for_category)
		plot_scatter_category(doc_vectors_for_category, ndim, args.output_dir, filename="docs")

	# 訓練データ全体で出現頻度が上位10000のものを取得
	common_words = cstm.get_high_freq_words(10000)
	plot_words(common_words, ndim, args.output_dir, filename="words")

	# 各文書についてfをプロット
	word_vector_pair = []
	for meta in common_words:
		word = meta[1]
		word_vector = np.asarray(meta[3], dtype=np.float32)
		word_vector_pair.append((word, word_vector))
	for doc_id in xrange(doc_vectors.shape[0]):
		plot_f(word_vector_pair, doc_id, doc_vectors[doc_id], args.output_dir)
	raise Exception()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model-dir", type=str, default="out")
	parser.add_argument("-o", "--output-dir", type=str, default="out/plot")
	parser.add_argument("-d", "--doc-id", type=int, default=0, help="文書ID")
	args = parser.parse_args()
	main(args)