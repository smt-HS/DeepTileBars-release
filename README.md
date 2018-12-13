# DeepTileBars-release

Implementation of [DeepTileBars: Visualizing Term Distribution of Neural Information Retrieval](https://arxiv.org/abs/1811.00606)


## Dependencies
```
pyspark
nltk
BeautifulSoup
keras
krovetzstemmer

```

## Running the model

### 0 Data Preparing

* Trained gensim word2vec model: `data/word2vec.100`. 

* Inverse Document Frequency (IDF) file: `data/term2idf.json`, which is essentially a dictionary storing the mapping `word -> idf`. 

* Query file: download from [TREC](https://trec.nist.gov/data/million.query/08/08.million-query-topics.10001-20000.gz), unzip and put it in the `data/08.million-query-topics` 

* LETOR-MQ2008 file: `./MQ2008/` is the folder downloaded from [Microsoft](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/#!letor-4-0)
 
### 1 Preprocessing
```bash
python preprocess.py
```



### 2 Extracting and cleaning documents
 ```bash
spark-submit --master [your-spark-cluster] extract_file.py /path/to/corpus /path/to/clean-file
```

### 3 TextTiling

__Warning:__ python3 users may need to fix a bug in NLTK follow this [post](https://github.com/nltk/nltk/pull/1863).

```bash
spark-submit --master [your-spark-cluster] texttiling.py /path/to/clean-file /path/to/segmented-file
```

### 4 Coloring
```bash
spark-submit --master [your-spark-cluster] text2img.py  /path/to/segmented-file /path/to/images
```

### 5 Run the model
```bash
python rank.py /path/to/images epochs
```
e.g.
```bash
python rank.py ./img 5
```

## Citation

If you are using this repo, please cite the following paper:


    @inproceedings{deeptilebars2018,
        title={DeepTileBars: Visualizing Term Distribution for Neural Information Retrieval},
        author={Tang, Zhiwen and Yang, Grace Hui},
        journal={AAAI 2019},
        year={2019}
    }