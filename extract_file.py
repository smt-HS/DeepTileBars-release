import os
import trecweb_parser
from bs4 import BeautifulSoup
from pyspark import SparkContext, SparkConf
import json
import sys

doc_set = set(json.load(open("data/docs.json")))


def generate_corpus_file_list(corpus_path):
    file_list = []
    corpus_path = os.path.abspath(corpus_path)
    for direc, subdirs, files in os.walk(corpus_path):
        for f in files:
            file_list.append(os.path.join(corpus_path, direc, f))
    return file_list


def extract(corpus_path, extracted_path):
    def extract_from_trecweb(file_path):
        for docno, dochdr, content in trecweb_parser.TrecWebParser(file_path, 'iso-8859-1'):
            if docno not in doc_set:
                continue

            soup = BeautifulSoup(content, 'html.parser')
            [s.extract() for s in soup('script')]
            [s.extract() for s in soup('style')]
            content = soup.get_text()

            with open(os.path.join(extracted_path, docno), "w") as output:
                print(content.encode(errors='replace').decode(), file=output)

    corpus_file_list = generate_corpus_file_list(corpus_path)
    conf = SparkConf().setAppName("extract_file")
    sc = SparkContext(conf=conf)

    os.makedirs(extracted_path, exist_ok=True)
    files = sc.parallelize(corpus_file_list, numSlices=128)
    files.map(extract_from_trecweb).collect()


if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2])
