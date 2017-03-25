# coding: utf-8
import argparse, sys, os, time, re, codecs, random

def split(path, args):
	filename = path.split("/")[-1]
	print filename
	sentences = []
	with codecs.open(path, "r", "utf-8") as f:
		for s in f:
			sentences.append(s)
	num_per_file = len(sentences) // args.split
	sentences = [sentences[i:i + num_per_file] for i in range(args.split)]
	for i, dataset in enumerate(sentences):
		with codecs.open("{}/{}_{}.txt".format(args.output_dir, filename, i), "w", "utf-8") as f:
			for sentence in dataset:
				f.write(sentence)

def main(args):
	try:
		os.mkdir(args.output_dir)
	except:
		pass
	assert args.input_dir is not None
	assert os.path.exists(args.input_dir)
	assert args.split > 0
	for filename in os.listdir(args.input_dir):
		if re.search(r".txt$", filename):
			split(args.input_dir + "/" + filename, args)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input-dir", type=str, default=None)
	parser.add_argument("-o", "--output-dir", type=str, default=None)
	parser.add_argument("-s", "--split", type=int, default=100)
	main(parser.parse_args())