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

# 文書グループ
# グループが存在しない場合は空にする
# 各グループ名から始まるテキストファイルは同じグループであるとみなす
# ただし、グループ関係はグラフプロット時の色分けにのみ使い、学習には用いない
# 1グループ1文書にする場合は空でよい
# 以下の例では geforce_00.txt geforce_01.txt などが同一グループになる
groups = [
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

def path_to_documents_for_group():
	return args.model_dir + "/documents_for_group.pickle"

def path_to_group_for_document():
	return args.model_dir +  "/group_for_document.pickle"

def add_documents(cstm):
	# 存在するファイルの読み込み
	filelist = os.listdir(args.document_dir)
	filelist.sort()

	# 文書にグループが存在しない場合
	if len(groups) == 0:
		for filename in filelist:
			if filename.endswith(".txt"):
				sys.stdout.write(stdout.CLEAR)
				sys.stdout.write("\rLoading {}".format(filename))
				sys.stdout.flush()
				cstm.add_document("{}/{}".format(args.document_dir, filename));
		return

	# 文書にグループが存在する場合
	documents_for_group = {}
	group_for_document = {}
	for group_name in groups:
		print "Loading group" + stdout.BOLD, group_name, stdout.END
		documents_for_group[group_name] = []
		for filename in filelist:
			if re.search(re.compile("^{}.*\.txt$".format(group_name)), filename):
				sys.stdout.write(stdout.CLEAR)
				sys.stdout.write("\rLoading {}".format(filename))
				sys.stdout.flush()
				doc_id = cstm.add_document("{}/{}".format(args.document_dir, filename));
				documents_for_group[group_name].append(doc_id)
				group_for_document[doc_id] = group_name
		sys.stdout.write(stdout.CLEAR)
		sys.stdout.write("\r")
		sys.stdout.flush()

	with open(path_to_documents_for_group(), "wb") as f:
		pickle.dump(documents_for_group, f)

	with open(path_to_group_for_document(), "wb") as f:
		pickle.dump(group_for_document, f)

def load_groups():
	documents_for_group = {}
	group_for_document = {}
	if os.path.exists(path_to_documents_for_group()):
		with open(path_to_documents_for_group(), "rb") as f:
			documents_for_group = pickle.load(f)

	if os.path.exists(path_to_group_for_document()):
		with open(path_to_group_for_document(), "rb") as f:
			group_for_document = pickle.load(f)

	return documents_for_group, group_for_document