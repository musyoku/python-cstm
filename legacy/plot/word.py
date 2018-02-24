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
sns.set(font=[fontfamily], font_scale=2)

def plot_words(words, ndim_vector, output_dir=None, filename="scatter"):
	with sns.axes_style("white", {"font.family": [fontfamily]}):
		for i in xrange(ndim_vector - 1):
			fig = pylab.gcf()
			fig.set_size_inches(45.0, 45.0)
			pylab.clf()
			for meta in words:
				word_id, word, count, vector = meta
				assert len(vector) == ndim_vector
				pylab.text(vector[i], vector[i + 1], word, fontsize=8)
			pylab.xlim(-4, 4)
			pylab.ylim(-4, 4)
			pylab.savefig("{}/{}_{}-{}.png".format(output_dir, filename, i, i + 1))

def main():
	mkdir(args.output_dir)
	assert os.path.exists(args.model_filename)
	cstm = model.cstm(args.model_filename)
	ndim = cstm.get_ndim_d()

	# 訓練データ全体で出現頻度が上位10000のものを取得
	common_words = cstm.get_high_freq_words(10000)	# 上位10,000個
	plot_words(common_words, ndim, args.output_dir, filename="words")

if __name__ == "__main__":
	main()