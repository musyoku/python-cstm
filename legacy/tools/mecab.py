# coding: utf-8
import argparse, codecs, sys, os, re
import MeCab

class stdout:
	BOLD = "\033[1m"
	END = "\033[0m"
	CLEAR = "\033[2K"

def main(args):
	# テキストファイルの追加
	files = []
	if args.input_dir is not None:
		assert os.path.exists(args.input_dir)
		fs = os.listdir(args.input_dir)
		for filename in fs:
			files.append(args.input_dir + "/" + filename)
	elif args.input_filename is not None:
		assert os.path.exists(args.input_filename)
		files.append(args.input_filename)
	else:
		raise Exception()

	try:
		os.mkdir(args.output_dir)
	except:
		pass

	for filepath in files:
		if filepath.endswith(".txt") == False:
			continue
		dataset = []
		sys.stdout.write(stdout.CLEAR)
		sys.stdout.write("\rprocessing {}".format(filepath))
		sys.stdout.flush()
		with codecs.open(filepath, "r", "utf-8") as f:
			tagger = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
			for i, sentence in enumerate(f):
				segmentation = ""
				sentence = re.sub(ur"\n", "", sentence)
				string = sentence.encode("utf-8")
				m = tagger.parseToNode(string)
				while m:
					segmentation += m.surface + " "
					m = m.next
				segmentation = re.sub(ur" $", "",  segmentation)
				segmentation = re.sub(ur"^ ", "",  segmentation)
				dataset.append(segmentation.decode("utf-8"))

		filename = filepath.split("/")[-1]
		with codecs.open(args.output_dir + "/" + filename, "w", "utf-8") as f:
			for sentence in dataset:
				f.write(sentence)
				f.write("\n")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--input-filename", type=str, default=None)
	parser.add_argument("-i", "--input-dir", type=str, default=None)
	parser.add_argument("-o", "--output-dir", type=str, default=None)
	args = parser.parse_args()
	main(args)