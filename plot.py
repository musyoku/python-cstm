# coding: utf-8
import argparse, sys, os, pylab
import model
import seaborn as sns
import numpy as np

# フォントをセット
# UbuntuならTakaoGothicなどが標準で入っている
if sys.platform == "darwin":
	fontfamily = "MS Gothic"
else:
	fontfamily = "TakaoGothic"
sns.set(font=[fontfamily], font_scale=1)

def mkdir(dir):
	assert dir is not None
	try:
		os.mkdir(dir)
	except:
		pass

def plot_kde(data, out_dir=None, filename="kde", color="Greens"):
	mkdir(out_dir)
	fig = pylab.gcf()
	fig.set_size_inches(16.0, 16.0)
	pylab.clf()
	bg_color = sns.color_palette(color, n_colors=256)[0]
	ax = sns.kdeplot(data[:, 0], data[:, 1], shade=True, cmap=color, n_levels=30, clip=[[-4, 4]] * 2)
	ax.set_axis_bgcolor(bg_color)
	kde = ax.get_figure()
	# pylab.xlim(-4, 4)
	# pylab.ylim(-4, 4)
	kde.savefig("{}/{}.png".format(out_dir, filename))

def plot_scatter(data, out_dir=None, filename="scatter", color="blue"):
	mkdir(out_dir)
	fig = pylab.gcf()
	fig.set_size_inches(16.0, 16.0)
	pylab.clf()
	pylab.scatter(data[:, 0], data[:, 1], s=20, marker="o", edgecolors="none", color=color)
	# pylab.xlim(-4, 4)
	# pylab.ylim(-4, 4)
	pylab.savefig("{}/{}.png".format(out_dir, filename))

def plot_scatter_category(data_for_category, ndim, out_dir=None, filename="scatter", color="blue"):
	mkdir(out_dir)
	markers = ["o", "v", "^", "<", ">"]
	palette = sns.color_palette("Set2", len(data_for_category))
	with sns.axes_style("white"):
		for i in xrange(ndim - 1):
			fig = pylab.gcf()
			fig.set_size_inches(16.0, 16.0)
			pylab.clf()
			for category, data in enumerate(data_for_category):
				pylab.scatter(data[:, i], data[:, i + 1], s=20, marker=markers[category % len(markers)], edgecolors="none", color=palette[category])
			# pylab.xlim(-4, 4)
			# pylab.ylim(-4, 4)
			pylab.savefig("{}/{}_{}-{}.png".format(out_dir, filename, i, i + 1))

def plot_words(words, ndim_vector, out_dir=None, filename="scatter"):
	mkdir(out_dir)
	with sns.axes_style("white", {"font.family": [fontfamily]}):
		for i in xrange(ndim_vector - 1):
			fig = pylab.gcf()
			fig.set_size_inches(16.0, 16.0)
			pylab.clf()
			for meta in words:
				word_id, word, count, vector = meta
				pylab.text(vector[i], vector[i + 1], word)
			pylab.xlim(-4, 4)
			pylab.ylim(-4, 4)
			pylab.savefig("{}/{}_{}-{}.png".format(out_dir, filename, i, i + 1))

def plot_f(words, doc_vectors, out_dir=None, filename="f"):
	with sns.axes_style("white", {"font.family": [fontfamily]}):
		collection = []
		for meta in words:
			word = meta[1]
			word_vector = np.asarray(meta[3], dtype=np.float32)
			collection.append((word, word_vector))
		for doc_id, doc_vector in enumerate(doc_vectors):
			fig = pylab.gcf()
			fig.set_size_inches(20.0, 10.0)
			pylab.clf()
			for meta in collection:
				word, word_vector = meta
				f = np.inner(word_vector, doc_vector)
				y = np.random.uniform(low=-5, high=5)
				pylab.text(f, y, word, fontsize=5)
			pylab.xlim(-20, 20)
			pylab.ylim(-5, 5)
			pylab.savefig("{}/{}_{}.png".format(out_dir, filename, doc_id))

def main(args):
	try:
		os.mkdir(args.output_dir)
	except:
		pass
	assert os.path.exists(args.model_dir)
	cstm = model.cstm()
	assert cstm.load(args.model_dir) == True
	print cstm.get_num_vocabulary(), "words"
	word_vectors = np.asarray(cstm.get_word_vectors(), dtype=np.float32)
	doc_vectors = np.asarray(cstm.get_doc_vectors(), dtype=np.float32)
	print doc_vectors
	print word_vectors

	ndim = cstm.get_ndim_vector()
	for i in xrange(ndim):
		print np.mean(word_vectors[:, i]), np.std(word_vectors[:, i])
	for i in xrange(ndim):
		print np.mean(doc_vectors[:, i]), np.std(doc_vectors[:, i])

	for doc_id in xrange(cstm.get_num_documents()):
		doc_vector = doc_vectors[doc_id]
		f = np.inner(word_vectors, doc_vector)
		print np.mean(f), np.std(f)

	raise Exception()
	
	# for i in xrange(ndim - 1):
	# 	plot_kde(word_vectors[:,i:], args.output_dir, filename="word_kde_{}-{}".format(i, i + 1))
	# 	plot_scatter(word_vectors[:,i:], args.output_dir, filename="word_scatter_{}-{}".format(i, i + 1))
	# 	plot_kde(doc_vectors[:,i:], args.output_dir, filename="doc_kde_{}-{}".format(i, i + 1))
	# 	plot_scatter(doc_vectors[:,i:], args.output_dir, filename="doc_scatter_{}-{}".format(i, i + 1))
		

	# 文書にカテゴリがある場合
	num_sections = 9
	doc_vectors_for_category = np.split(doc_vectors, num_sections)
	plot_scatter_category(doc_vectors_for_category, ndim, args.output_dir, filename="doc_for_category")

	common_words = cstm.get_high_freq_words(10000)
	plot_words(common_words, ndim, args.output_dir, filename="words")
	raise Exception()

	plot_f(common_words, doc_vectors, args.output_dir)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model-dir", type=str, default="out")
	parser.add_argument("-o", "--output-dir", type=str, default="out/plot")
	main(parser.parse_args())