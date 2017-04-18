# coding: utf-8
import argparse, codecs, os, sys, re, pylab
from gensim import models
from sets import Set
from gensim.models.doc2vec import LabeledSentence
import numpy as np

def main(args):
	assert os.path.exists(args.model_filename)
	model = models.Doc2Vec.load(args.model_filename)
	for i in model.most_similar(positive=u"けもフレ"):
	    print i[0].encode(sys.stdout.encoding), i[1]

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--model-filename", type=str, default="out/doc2vec.model")
	parser.add_argument("-w", "--word", type=str, default="your_positive_word")
	args = parser.parse_args()
	main(args)
