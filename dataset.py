# coding: utf-8
import sys, os, re
import model
import numpy as np
import pickle
from args import args

class stdout:
	BOLD = "\033[1m"
	END = "\033[0m"
	CLEAR = "\033[2K"

# 文書カテゴリ
# カテゴリが存在しない場合は空にする
# 各カテゴリ名から始まるテキストファイルは同じカテゴリであるとみなす
# ただしカテゴリの情報はグラフプロット時の色分けにのみ使い、学習には用いない
# （カテゴリごとに文書ベクトルが近い位置に集まってくるかどうかを見るため）
# 以下の例では geforce_00.txt geforce_01.txt などが同一カテゴリになる
categories = [
	"geforce",
	"gochiusa",
	"imas",
	"kemono",
	"macbook",
	"monst",
	"pad",
	"tekketsu",
	"win10",
]
# 空にするとargs.document_dir内のファイルを全て読み込む
# 通常は空にしておく
# categories = []

def path_to_documents_for_category():
	model_dir = "/".join(args.model_filename.split("/")[:-1])
	return model_dir + "/documents_for_category.pickle"

def path_to_category_for_document():
	model_dir = "/".join(args.model_filename.split("/")[:-1])
	return model_dir +  "/category_for_document.pickle"

def add_documents(cstm):
	# 存在するファイルの読み込み
	filelist = os.listdir(args.document_dir)
	filelist.sort()

	# 文書にカテゴリが存在しない場合
	if len(categories) == 0:
		for filename in filelist:
			if filename.endswith(".txt"):
				sys.stdout.write(stdout.CLEAR)
				sys.stdout.write("\rLoading {}".format(filename))
				sys.stdout.flush()
				cstm.add_document("{}/{}".format(args.document_dir, filename));
		sys.stdout.write(stdout.CLEAR)
		sys.stdout.write("\r")
		sys.stdout.flush()
		return

	# 文書にカテゴリが存在する場合
	documents_for_category = {}
	category_for_document = {}
	for category_name in categories:
		print "Loading category" + stdout.BOLD, category_name, stdout.END
		documents_for_category[category_name] = []
		for filename in filelist:
			if re.search(re.compile("^{}.*\.txt$".format(category_name)), filename):
				sys.stdout.write(stdout.CLEAR)
				sys.stdout.write("\rLoading {}".format(filename))
				sys.stdout.flush()
				doc_id = cstm.add_document("{}/{}".format(args.document_dir, filename));
				documents_for_category[category_name].append(doc_id)
				category_for_document[doc_id] = category_name
		sys.stdout.write(stdout.CLEAR)
		sys.stdout.write("\r")
		sys.stdout.flush()

	with open(path_to_documents_for_category(), "wb") as f:
		pickle.dump(documents_for_category, f)

	with open(path_to_category_for_document(), "wb") as f:
		pickle.dump(category_for_document, f)

def load_categories():
	documents_for_category = {}
	category_for_document = {}
	if os.path.exists(path_to_documents_for_category()):
		with open(path_to_documents_for_category(), "rb") as f:
			documents_for_category = pickle.load(f)

	if os.path.exists(path_to_category_for_document()):
		with open(path_to_category_for_document(), "rb") as f:
			category_for_document = pickle.load(f)

	return documents_for_category, category_for_document