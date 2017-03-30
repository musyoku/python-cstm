import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model-dir", type=str, default="out")
parser.add_argument("-o", "--output-dir", type=str, default="out/plot")
parser.add_argument("-t", "--document-dir", type=str, default="documents")
parser.add_argument("-d", "--ndim-d", type=int, default=20)
args = parser.parse_args()