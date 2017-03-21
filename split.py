# coding: utf-8
import argparse, sys, os, time, re, codecs, random

def main(args):
	try:
		os.mkdir(args.output_dir)
	except:
		pass
	assert args.input_filename is not None
	assert os.path.exists(args.input_filename)
	assert args.split > 0
	sentences = []
	with codecs.open(args.input_filename, "r", "utf-8") as f:
		for s in f:
			sentences.append(s)
	num_per_file = len(sentences) // args.split
	sentences = [sentences[i:i + num_per_file] for i in range(args.split)]
	for i, dataset in enumerate(sentences):
		with codecs.open("{}/{}.txt".format(args.output_dir, i), "w", "utf-8") as f:
			for sentence in dataset:
				f.write(sentence)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input-filename", type=str, default=None)
	parser.add_argument("-o", "--output-dir", type=str, default=None)
	parser.add_argument("-s", "--split", type=int, default=20)
	main(parser.parse_args())