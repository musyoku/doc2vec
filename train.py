# coding: utf-8
import argparse, codecs, os, sys, re, pylab
from gensim import models
from sets import Set
from gensim.models.doc2vec import LabeledSentence
import seaborn as sns
import numpy as np

class stdout:
	BOLD = "\033[1m"
	END = "\033[0m"
	CLEAR = "\033[2K"

# フォントをセット
# UbuntuならTakaoGothicなどが標準で入っている
if sys.platform == "darwin":
	fontfamily = "MS Gothic"
else:
	fontfamily = "TakaoGothic"
sns.set(font=[fontfamily], font_scale=2)

def train(args):
	model_dir = "/".join(args.model_filename.split("/")[:-1])
	try:
		os.mkdir(model_dir)
	except:
		pass

	# 読み込み
	filelist = os.listdir(args.document_dir)
	filelist.sort()
	sentences = []
	for filename in filelist:
		if re.search(r".txt$", filename):
			sys.stdout.write(stdout.CLEAR)
			sys.stdout.write("\rLoading {}".format(filename))
			path = "{}/{}".format(args.document_dir, filename)
			with codecs.open(path, "r", "utf-8") as f:
				whole_words = []
				for sentence in f:
					words = sentence.split(" ")
					for word in words:
						whole_words.append(word)
				sentences.append(LabeledSentence(words=whole_words, tags=[filename]))
	
	model = models.Doc2Vec(sentences, dm=1, size=10, window=15, alpha=.025, min_alpha=.025, min_count=1, sample=1e-6)

	for epoch in range(2000):
		print(stdout.CLEAR)
		print("Epoch: {}".format(epoch + 1))
		model.train(sentences)
		model.alpha -= (0.025 - 0.0001) / 19
		model.min_alpha = model.alpha
		model.save(args.model_filename)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model-filename", type=str, default="out/doc2vec.model")
	parser.add_argument("-d", "--document-dir", type=str, default=None)
	args = parser.parse_args()
	train(args)