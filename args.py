# coding: utf-8
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model-dir", type=str, default="model")
parser.add_argument("-o", "--output-dir", type=str, default="model/out")
parser.add_argument("-d", "--document-dir", type=str, default="documents")
parser.add_argument("-dim", "--ndim-d", type=int, default=20)
parser.add_argument("-thread", "--num-thread", type=int, default=1)
parser.add_argument("-ignore", "--ignore-count", type=int, default=0, help="これ以下の出現頻度の単語は学習しない")
args = parser.parse_args()