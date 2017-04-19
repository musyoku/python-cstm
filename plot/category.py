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

def plot_scatter_category(vectors_for_category, output_dir=None, filename="scatter", color="blue"):
	ndim = vectors_for_category[0].shape[1]
	markers = ["o", "v", "^", "<", ">"]
	palette = sns.color_palette("Set2", len(vectors_for_category))
	with sns.axes_style("white"):
		for i in xrange(ndim - 1):
			fig = pylab.gcf()
			fig.set_size_inches(16.0, 16.0)
			pylab.clf()
			for category_id, vectors in enumerate(vectors_for_category):
				assert vectors.shape[1] == ndim
				pylab.scatter(vectors[:, i], vectors[:, i + 1], s=30, marker=markers[category_id % len(markers)], edgecolors="none", color=palette[category_id])
			# pylab.xlim(-4, 4)
			# pylab.ylim(-4, 4)
			pylab.savefig("{}/{}_{}-{}.png".format(output_dir, filename, i, i + 1))

def main():
	mkdir(args.output_dir)
	assert os.path.exists(args.model_filename)
	cstm = model.cstm(args.model_filename)
	ndim = cstm.get_ndim_d()
	# ベクトルを取得
	word_vectors = np.asarray(cstm.get_word_vectors(), dtype=np.float32)
	doc_vectors = np.asarray(cstm.get_doc_vectors(), dtype=np.float32)

	# 文書にカテゴリがある場合
	# ファイル名の接頭語でカテゴリを判断
	categories = [
		"geforce",
		"gochiusa",
		"imas",
		"kemono",
		"macbook",
		"monst",
		"pad",
		"tekketsu",
		"win10",
	]

	doc_filenames = cstm.get_doc_filenames()
	doc_vectors_for_category = []
	for category_id, category_name in enumerate(categories):
		doc_vectors_for_category.append([])
		for filename in doc_filenames:
			if filename.startswith(category_name):
				print category_name
				doc_id = cstm.get_doc_id_by_filename(filename)
				assert doc_id >= 0 and doc_id < len(doc_vectors)
				doc_vectors_for_category[category_id].append(doc_vectors[doc_id])
	doc_vectors_for_category = np.asanyarray(doc_vectors_for_category)
	plot_scatter_category(doc_vectors_for_category, args.output_dir, filename="category")

if __name__ == "__main__":
	main()