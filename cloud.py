# coding: utf-8
import argparse, codecs, os, sys, re, pylab
from gensim import models
from sets import Set
from gensim.models.doc2vec import LabeledSentence
import seaborn as sns
import numpy as np
from wordcloud import WordCloud, ImageColorGenerator

def color_func_1(word, font_size, position, orientation, random_state=None, **kwargs):
    colors = (
        "rgb(192, 57, 43)",
        "rgb(231, 76, 60)",
        "rgb(243, 156, 18)",
        "rgb(241, 196, 15)",
        "rgb(142, 68, 173)",
        "rgb(155, 89, 182)",
        "rgb(202, 44, 104)",
        "rgb(234, 76, 136)",
        "rgb(44, 62, 80)",
        "rgb(52, 73, 94)",
        "rgb(41, 128, 185)",
        "rgb(52, 152, 219)",
        "rgb(52, 152, 219)",
        "rgb(22, 160, 133)",
        "rgb(26, 188, 156)",
        )
    index = random.randint(0, len(colors) - 1)
    return colors[index]

def color_func_2(word, font_size, position, orientation, random_state=None, **kwargs):
    colors = (
        "rgb(251, 115, 116)",
        "rgb(0, 163, 136)",
        "rgb(255, 92, 157)",
        "rgb(121, 191, 161)",
        "rgb(245, 163, 82)",
        )
    index = random.randint(0, len(colors) - 1)
    return colors[index]

def color_func_3(word, font_size, position, orientation, random_state=None, **kwargs):
    colors = (
        "rgb(35, 75, 113)",
        "rgb(74, 133, 189)",
        "rgb(191, 148, 79)",
        "rgb(128, 193, 255)",
        "rgb(227, 184, 114)",
        )
    index = random.randint(0, len(colors) - 1)
    return colors[index]

def color_func_4(word, font_size, position, orientation, random_state=None, **kwargs):
    colors = (
        "rgb(194, 193, 165)",
        "rgb(162, 148, 104)",
        "rgb(61, 102, 97)",
        "rgb(28, 52, 60)",
        "rgb(117, 148, 131)",
        )
    index = random.randint(0, len(colors) - 1)
    return colors[index]


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
	# docs
	doc_vectors = np.asanyarray(doc_vectors)

	# inner product
	doc_vector = doc_vectors[args.doc_id]
	inner = {}
	for word in word_count:
		count = word_count[word]
		if count <= args.ignore_count:
			continue
		word_vector = model[word]
		f = np.inner(word_vector, doc_vector)
		inner[word] = f
		
	wordcloud = WordCloud(
		background_color="white",
		font_path=args.font_path, 
		width=args.width, 
		height=args.height, 
		max_words=max_count, 
		max_font_size=args.max_font_size).generate_from_frequencies(inner)
	color_funcs = [None, color_func_1, color_func_2, color_func_3, color_func_4]
	color_func = color_funcs[args.color]
	wordcloud.recolor(color_func=color_func)
	wordcloud.to_file("{}/cloud.png".format(args.output_dir))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model-filename", type=str, default="out/doc2vec.model")
	parser.add_argument("-o", "--output-dir", type=str, default="out")
	parser.add_argument("-d", "--document-dir", type=str, default=None)
	parser.add_argument("-doc", "--doc-id", type=int, default=0)
	parser.add_argument("--width", type=int, default=1440, help="クラウドの幅.")
	parser.add_argument("--height", type=int, default=1080, help="クラウドの高さ.")
	parser.add_argument("--color", type=int, default=1, help="クラウドのcolor_func番号.")
	parser.add_argument("-fsize", "--max-font-size", type=int, default=300, help="最大フォントサイズ.")
	parser.add_argument("-max", "--max-num-word", type=int, default=500, help="fの値が高い順にいくつの単語をプロットするか.")
	parser.add_argument("-font", "--font-path", type=str, default=None, help="フォントのパス.")
	parser.add_argument("-min", "--min-occurence", type=int, default=20, help="これ以下の出現回数の単語はプロットしない.")
	args = parser.parse_args()
	plot(args)
