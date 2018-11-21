from pyspark import SparkContext, SparkConf
import os
from nltk.tokenize import TextTilingTokenizer
import sys


def texttiling(source_direc, dest_direc):
    conf = SparkConf().setAppName("text_tiling")
    sc = SparkContext(conf=conf)

    source_direc = "/Users/zhiwentang/DeepTileBars/dirty_extracted_docs_after_clean"
    dest_direc = "/Users/zhiwentang/DeepTileBars/segment_docs_k_6/"

    files = sc.parallelize(os.listdir(source_direc))

    tokenizer = TextTilingTokenizer(k=6)

    def texttiling_doc(file_name):
        content = open(os.path.join(source_direc, file_name)).read()
        doc_id = file_name.split(".")[0]
        try:
            segments = tokenizer.tokenize(content)
            output = '\n\n'.join([segment.replace("\n", "\t") for segment in segments])
        except:
            output = content.replace("\n", "\t")

        with open(os.path.join(dest_direc, doc_id), "w") as f:
            print(output, file=f)

    files.map(texttiling_doc).collect()


if __name__ == "__main__":
    texttiling(sys.argv[1], sys.argv[2])
