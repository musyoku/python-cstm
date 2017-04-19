# coding: utf-8
import argparse, sys, os, pylab, random, argparse, codecs
import seaborn as sns
import numpy as np
from wordcloud import WordCloud, ImageColorGenerator

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
	assert os.path.exists(args.document_filename)

	freq = {}
	with codecs.open(args.document_filename, "r", "utf-8") as f:
		for sentence in f:
			words = sentence.split(" ")
			for word in words:
				if len(word) == 0:
					continue
				if word not in freq:
					freq[word] = 0;
				freq[word] += 1

	ranking = sorted(freq.items(), key=lambda x: -x[1])	# sortedが昇順なのでマイナスを掛ける
	for i in xrange(20):
		word, count = ranking[i]
		print word.encode(sys.stdout.encoding), count

	wordcloud = WordCloud(
		background_color="white",
		font_path=args.font_path, 
		width=args.width, 
		height=args.height, 
		max_words=args.max_num_word, 
		max_font_size=args.max_font_size).generate_from_frequencies(freq)
	color_funcs = [None, color_func_1, color_func_2, color_func_3, color_func_4]
	color_func = color_funcs[args.color]
	wordcloud.recolor(color_func=color_func)

	filename = args.document_filename.split("/")[-1]
	wordcloud.to_file("{}/cloud_freq_{}.png".format(args.output_dir, filename))
				
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-o", "--output-dir", type=str, default="../out")
	parser.add_argument("-doc", "--document-filename", type=str, default=None, help="文書ファイルのパス")
	parser.add_argument("--width", type=int, default=1440, help="クラウドの幅.")
	parser.add_argument("--height", type=int, default=1080, help="クラウドの高さ.")
	parser.add_argument("--color", type=int, default=1, help="クラウドのcolor_func番号.")
	parser.add_argument("-fsize", "--max-font-size", type=int, default=200, help="最大フォントサイズ.")
	parser.add_argument("-max", "--max-num-word", type=int, default=500, help="fの値が高い順にいくつの単語をプロットするか.")
	parser.add_argument("-font", "--font-path", type=str, default=None, help="フォントのパス.")
	parser.add_argument("-min", "--min-occurence", type=int, default=20, help="これ以下の出現回数の単語はプロットしない.")
	args = parser.parse_args()
	main(args)