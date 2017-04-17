import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model-dir", type=str, default="out")
parser.add_argument("-o", "--output-dir", type=str, default="out/plot")
parser.add_argument("-d", "--document-dir", type=str, default="documents")
parser.add_argument("-dim", "--ndim-d", type=int, default=20)
parser.add_argument("-thread", "--num-thread", type=int, default=1)
args = parser.parse_args()