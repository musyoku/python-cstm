# coding: utf-8
import argparse, sys, os, pylab, random, argparse
import seaborn as sns
import numpy as np
from wordcloud import WordCloud, ImageColorGenerator
sys.path.append(os.path.split(os.getcwd())[0])
import model

def color_func_1(word, font_size, position, orientation, random_state=None, **kwargs):
    colors = (
        "rgb(192, 57, 43)",
        "rgb(231, 76, 60)",
        "rgb(243, 156, 18)",
        "rgb(241, 196, 15)",
        "rgb(142, 68, 173)",
        "rgb(155, 89, 182)",
        "rgb(202, 44, 104)",
        "rgb(234, 76, 136)",
        "rgb(44, 62, 80)",
        "rgb(52, 73, 94)",
        "rgb(41, 128, 185)",
        "rgb(52, 152, 219)",
        "rgb(52, 152, 219)",
        "rgb(22, 160, 133)",
        "rgb(26, 188, 156)",
        )
    index = random.randint(0, len(colors) - 1)
    return colors[index]

def color_func_2(word, font_size, position, orientation, random_state=None, **kwargs):
    colors = (
        "rgb(251, 115, 116)",
        "rgb(0, 163, 136)",
        "rgb(255, 92, 157)",
        "rgb(121, 191, 161)",
        "rgb(245, 163, 82)",
        )
    index = random.randint(0, len(colors) - 1)
    return colors[index]

def color_func_3(word, font_size, position, orientation, random_state=None, **kwargs):
    colors = (
        "rgb(35, 75, 113)",
        "rgb(74, 133, 189)",
        "rgb(191, 148, 79)",
        "rgb(128, 193, 255)",
        "rgb(227, 184, 114)",
        )
    index = random.randint(0, len(colors) - 1)
    return colors[index]

def color_func_4(word, font_size, position, orientation, random_state=None, **kwargs):
    colors = (
        "rgb(194, 193, 165)",
        "rgb(162, 148, 104)",
        "rgb(61, 102, 97)",
        "rgb(28, 52, 60)",
        "rgb(117, 148, 131)",
        )
    index = random.randint(0, len(colors) - 1)
    return colors[index]

def mkdir(target):
	assert target is not None
	try:
		os.mkdir(target)
	except:
		pass

def main(args):
	assert args.font_path is not None
	mkdir(args.output_dir)
	assert os.path.exists(args.model_filename)
	cstm = model.cstm(args.model_filename)
	ndim = cstm.get_ndim_d()
	# ベクトルを取得
	doc_vectors = np.asarray(cstm.get_doc_vectors(), dtype=np.float32)
	assert args.doc_id < doc_vectors.shape[0]
	doc_vector = doc_vectors[args.doc_id]

	# 各単語についてfを計算
	dic = {}
	words = cstm.get_words()
	for meta in words:
		word = meta[1]
		count = meta[2]
		if count < args.min_occurence:
			continue
		word_vector = np.asarray(meta[3], dtype=np.float32)
		f = np.inner(word_vector, doc_vector)
		dic[word] = f
	dic = sorted(dic.items(), key=lambda x: -x[1])	# sortedが昇順なのでマイナスを掛ける

	max_count = min(args.max_num_word, len(dic))
	dic = dict(dic[:max_count])

	wordcloud = WordCloud(
		background_color="white",
		font_path=args.font_path, 
		width=args.width, 
		height=args.height, 
		max_words=max_count, 
		max_font_size=args.max_font_size).generate_from_frequencies(dic)
	color_funcs = [None, color_func_1, color_func_2, color_func_3, color_func_4]
	color_func = color_funcs[args.color]
	wordcloud.recolor(color_func=color_func)
	wordcloud.to_file("{}/cloud_f_{}.png".format(args.output_dir, args.doc_id))
				
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model-filename", type=str, default="../model/cstm.model")
	parser.add_argument("-o", "--output-dir", type=str, default="../out")
	parser.add_argument("-doc", "--doc-id", type=int, default=0, help="文書ID")
	parser.add_argument("--width", type=int, default=1440, help="クラウドの幅.")
	parser.add_argument("--height", type=int, default=1080, help="クラウドの高さ.")
	parser.add_argument("--color", type=int, default=1, help="クラウドのcolor_func番号.")
	parser.add_argument("-fsize", "--max-font-size", type=int, default=200, help="最大フォントサイズ.")
	parser.add_argument("-max", "--max-num-word", type=int, default=500, help="fの値が高い順にいくつの単語をプロットするか.")
	parser.add_argument("-font", "--font-path", type=str, default=None, help="フォントのパス.")
	parser.add_argument("-min", "--min-occurence", type=int, default=20, help="これ以下の出現回数の単語はプロットしない.")
	args = parser.parse_args()
	main(args)