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

def plot_scatter_category(data_for_category, ndim, output_dir=None, filename="scatter", color="blue"):
	markers = ["o", "v", "^", "<", ">"]
	palette = sns.color_palette("Set2", len(data_for_category))
	with sns.axes_style("white"):
		for i in xrange(ndim - 1):
			fig = pylab.gcf()
			fig.set_size_inches(16.0, 16.0)
			pylab.clf()
			for category, data in enumerate(data_for_category):
				pylab.scatter(data[:, i], data[:, i + 1], s=30, marker=markers[category % len(markers)], edgecolors="none", color=palette[category])
			# pylab.xlim(-4, 4)
			# pylab.ylim(-4, 4)
			pylab.savefig("{}/{}_{}-{}.png".format(output_dir, filename, i, i + 1))

def plot_words(words, ndim_vector, output_dir=None, filename="scatter"):
	with sns.axes_style("white", {"font.family": [fontfamily]}):
		for i in xrange(ndim_vector - 1):
			fig = pylab.gcf()
			fig.set_size_inches(45.0, 45.0)
			pylab.clf()
			for meta in words:
				word, vector = meta
				pylab.text(vector[i], vector[i + 1], word, fontsize=5)
			# pylab.xlim(-4, 4)
			# pylab.ylim(-4, 4)
			pylab.savefig("{}/{}_{}-{}.png".format(output_dir, filename, i, i + 1))

def plot_scatter_docs(vectors, output_dir=None, filename="scatter", color="blue"):
	ndim = vectors.shape[1]
	with sns.axes_style("white"):
		for i in xrange(ndim - 1):
			fig = pylab.gcf()
			fig.set_size_inches(16.0, 16.0)
			pylab.clf()
			pylab.scatter(vectors[:, i], vectors[:, i + 1], s=30, edgecolors="none")
			# pylab.xlim(-4, 4)
			# pylab.ylim(-4, 4)
			pylab.savefig("{}/{}_{}-{}.png".format(output_dir, filename, i, i + 1))

def plot(args):
	assert os.path.exists(args.model_filename)
	model = models.Doc2Vec.load(args.model_filename)
	filelist = os.listdir(args.document_dir)
	filelist.sort()
	vectors = []
	word_count = {}
	for filename in filelist:
		if re.search(r".txt$", filename):
			vectors.append(model.docvecs[filename])
			sys.stdout.write(stdout.CLEAR)
			sys.stdout.write("\rLoading {}".format(filename))
			path = "{}/{}".format(args.document_dir, filename)
			with codecs.open(path, "r", "utf-8") as f:
				for sentence in f:
					words = sentence.split(" ")
					for word in words:
						if word not in word_count:
							word_count[word] = 0
						word_count[word] += 1
	# words
	collection = []
	for word in word_count:
		count = word_count[word]
		if count <= args.ignore_count:
			continue
		vector = model[word]
		collection.append((word, vector))
	# plot_words(collection, 4, output_dir=args.output_dir, filename="words")

	# docs
	doc_vectors = np.asanyarray(doc_vectors)
	# vectors_for_category = np.split(doc_vectors, 9)
	# plot_scatter_category(doc_vectors, 4, output_dir=args.output_dir, filename="documents")
	plot_scatter_docs(doc_vectors, output_dir=args.output_dir, filename="documents")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model-filename", type=str, default="out/doc2vec.model")
	parser.add_argument("-o", "--output-dir", type=str, default="out")
	parser.add_argument("-d", "--document-dir", type=str, default=None)
	parser.add_argument("-ignore", "--ignore-count", type=int, default=10)
	args = parser.parse_args()
	plot(args)
