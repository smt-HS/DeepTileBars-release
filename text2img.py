import os
from pyspark import SparkContext, SparkConf
from collections import Counter
import json
import utils
import math
import numpy as np
from gensim.models import Word2Vec
import sys


def text2img(segment_direc, img_direc):
    conf = SparkConf().setAppName("text2img")
    sc = SparkContext(conf=conf)
    query_map = sc.broadcast(json.load(open("data/query_map.json")))
    term2idf = sc.broadcast(json.load(open("data/term2idf.json")))
    word2vec_model = Word2Vec.load("data/word2vec.100")
    word2vec = sc.broadcast(word2vec_model.wv)
    del word2vec_model

    def transform2img2(line):
        max_ql, max_bl = 9, 30  # 9 query terms, 30 blocks
        # each query has no greater than 5 terms
        # about 90% docs have no greater than 30 blocks

        parts = line.strip().split()
        qid, doc = parts[0], parts[2]

        query = query_map.value[qid]
        path = os.path.join(segment_direc, doc)
        if not os.path.exists(path):
            print(qid, doc)
            return
        segments = open(path).read().split("\n\n")

        query_terms = query.split()
        query_terms = [term for term in query_terms if len(term) > 0]

        matrix = np.zeros((max_ql, max_bl, 2))

        ql = len(query_terms)
        bl = len(segments)

        if bl > max_bl:
            segments[max_bl - 1] = "\t".join(segments[max_bl - 1:])
            bl = max_bl

        for j in range(bl):
            block = segments[j]
            # if len(block) < 1:
            #    continue
            # doc_temrs = index.tokenize(block)
            doc_temrs = utils.tokenize(block)
            tf = Counter(doc_temrs)
            doc_vecs = [word2vec.value[term] for term in doc_temrs if term in word2vec.value.vocab]
            for i in range(ql):
                # if query_terms[i] not in term2idf.value:
                #    continue

                if tf[query_terms[i]] == 0:
                    matrix[i, j, 0] = -10
                    if query_terms[i] not in word2vec.value.vocab or len(doc_vecs) == 0:
                        matrix[i, j, 1] = 0
                    else:
                        query_vec = word2vec.value[query_terms[i]]
                        diffs = [np.linalg.norm(doc_vec - query_vec) for doc_vec in doc_vecs]
                        matrix[i, j, 1] = math.exp(-min(diffs))
                else:
                    if query_terms[i] in term2idf.value:
                        matrix[i, j, 0] = tf[query_terms[i]] * term2idf.value[query_terms[i]]
                    else:
                        matrix[i, j, 0] = tf[query_terms[i]]
                    matrix[i, j, 1] = 1

        os.makedirs(os.path.join(img_direc, qid), exist_ok=True)

        np.save(os.path.join(img_direc, qid, doc), matrix)

    tuples = sc.textFile("qrels/MQ2008.txt", minPartitions=32)

    tuples.map(transform2img2).collect()


if __name__ == "__main__":
    text2img(sys.argv[1], sys.argv[2])
