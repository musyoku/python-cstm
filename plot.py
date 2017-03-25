# coding: utf-8
import argparse, sys, os, pylab
import model
import seaborn as sns
import numpy as np

def plot_kde(data, dir=None, filename="kde", color="Greens"):
	assert dir is not None
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(16.0, 16.0)
	pylab.clf()
	bg_color = sns.color_palette(color, n_colors=256)[0]
	ax = sns.kdeplot(data[:, 0], data[:, 1], shade=True, cmap=color, n_levels=30, clip=[[-4, 4]] * 2)
	ax.set_axis_bgcolor(bg_color)
	kde = ax.get_figure()
	pylab.xlim(-4, 4)
	pylab.ylim(-4, 4)
	kde.savefig("{}/{}.png".format(dir, filename))

def plot_scatter(data, dir=None, filename="scatter", color="blue"):
	if dir is None:
		raise Exception()
	try:
		os.mkdir(dir)
	except:
		pass
	fig = pylab.gcf()
	fig.set_size_inches(16.0, 16.0)
	pylab.clf()
	pylab.scatter(data[:, 0], data[:, 1], s=20, marker="o", edgecolors="none", color=color)
	pylab.xlim(-4, 4)
	pylab.ylim(-4, 4)
	pylab.savefig("{}/{}.png".format(dir, filename))

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
	print np.mean(word_vectors[:, 0]), np.std(word_vectors[:, 0])
	print np.mean(word_vectors[:, 1]), np.std(word_vectors[:, 1])
	plot_kde(word_vectors, args.output_dir, filename="word_kde")
	plot_scatter(word_vectors, args.output_dir, filename="word_scatter")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model-dir", type=str, default="out")
	parser.add_argument("-o", "--output-dir", type=str, default="out")
	main(parser.parse_args())