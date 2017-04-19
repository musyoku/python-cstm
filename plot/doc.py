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

def plot_kde(data, output_dir=None, filename="kde", color="Greens"):
	fig = pylab.gcf()
	fig.set_size_inches(16.0, 16.0)
	pylab.clf()
	bg_color = sns.color_palette(color, n_colors=256)[0]
	ax = sns.kdeplot(data[:, 0], data[:, 1], shade=True, cmap=color, n_levels=30, clip=[[-4, 4]] * 2)
	ax.set_axis_bgcolor(bg_color)
	kde = ax.get_figure()
	# pylab.xlim(-4, 4)
	# pylab.ylim(-4, 4)
	kde.savefig("{}/{}.png".format(output_dir, filename))

def plot_scatter(data, output_dir=None, filename="scatter", color="blue"):
	fig = pylab.gcf()
	fig.set_size_inches(16.0, 16.0)
	pylab.clf()
	pylab.scatter(data[:, 0], data[:, 1], s=20, marker="o", edgecolors="none", color=color)
	# pylab.xlim(-4, 4)
	# pylab.ylim(-4, 4)
	pylab.savefig("{}/{}.png".format(output_dir, filename))

def main():
	mkdir(args.output_dir)
	assert os.path.exists(args.model_filename)
	cstm = model.cstm(args.model_filename)
	ndim_d = cstm.get_ndim_d()

	# ベクトルを取得
	doc_vectors = np.asarray(cstm.get_doc_vectors(), dtype=np.float32)

	for i in xrange(ndim_d - 1):
		plot_kde(doc_vectors[:,i:], args.output_dir, filename="doc_kde_{}-{}".format(i, i + 1))
		plot_scatter(doc_vectors[:,i:], args.output_dir, filename="doc_scatter_{}-{}".format(i, i + 1))

if __name__ == "__main__":
	main()