# DeepTileBars-release

Implementation of [DeepTileBars: Visualizing Term Distribution of Neural Information Retrieval](https://arxiv.org/abs/1811.00606)


## Dependencies
```
pyspark
nltk
BeautifulSoup
keras

```


### 1 Preprocessing



### 2 Extracting and cleaning documents
 ```bash
spark-submit --master [your-spark-cluster] extract_file.py /path/to/corpus /path/to/clean-file
```

### 3 TextTiling

__Warning:__ python3 users may need to fix a bug in NLTK follow this [post](https://github.com/nltk/nltk/pull/1863).

```bash
spark-submit --master [your-spark-cluster] texttiling.py /path/to/clean-file /path/to/segmented-file
```
