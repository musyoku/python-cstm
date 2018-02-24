# coding: utf-8
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model-filename", type=str, default="../model/cstm.model")
parser.add_argument("-o", "--output-dir", type=str, default="../out")
parser.add_argument("-doc", "--doc-id", type=int, default=0)
args = parser.parse_args()