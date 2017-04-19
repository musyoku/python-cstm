# coding: utf-8
import argparse, sys, os, pylab
import seaborn as sns
import numpy as np
sys.path.append(os.path.split(os.getcwd())[0])
import model
from args import args

def mkdir(target):
	assert target is not None
	try:
		os.mkdir(target)
	except:
		pass
		
# フォントをセット
# UbuntuならTakaoGothicなどが標準で入っている
if sys.platform == "darwin":
	fontfamily = "MS Gothic"
else:
	fontfamily = "TakaoGothic"
sns.set(font=[fontfamily], font_scale=1)

def plot_f(words, doc_id, doc_vector, output_dir=None, filename="f"):
	with sns.axes_style("white", {"font.family": [fontfamily]}):
		fig = pylab.gcf()
		fig.set_size_inches(40.0, 40.0)
		pylab.clf()
		for meta in words:
			word, word_vector = meta
			f = np.inner(word_vector, doc_vector)
			y = np.random.uniform(low=-5, high=5)
			pylab.text(f, y, word, fontsize=10)
		pylab.xlim(-20, 20)
		pylab.ylim(-5, 5)
		pylab.savefig("{}/{}_{}.png".format(output_dir, filename, doc_id))

def main():
	mkdir(args.output_dir)
	assert os.path.exists(args.model_filename)
	cstm = model.cstm(args.model_filename)
	ndim = cstm.get_ndim_d()
	# ベクトルを取得
	word_vectors = np.asarray(cstm.get_word_vectors(), dtype=np.float32)
	doc_vectors = np.asarray(cstm.get_doc_vectors(), dtype=np.float32)

	# 訓練データ全体で出現頻度が上位10000のものを取得
	common_words = cstm.get_high_freq_words(10000)

	# 各文書についてfをプロット
	word_vector_pair = []
	for meta in common_words:
		word = meta[1]
		word_vector = np.asarray(meta[3], dtype=np.float32)
		word_vector_pair.append((word, word_vector))
	plot_f(word_vector_pair, args.doc_id, doc_vectors[args.doc_id], args.output_dir)

if __name__ == "__main__":
	main()