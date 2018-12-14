import krovetzstemmer

char_set = set([str(i) for i in range(10)])
char_set.update([chr(i) for i in range(ord('a'), ord('z') + 1)])
char_set.update([chr(i) for i in range(ord('A'), ord('Z') + 1)])

stopword_set = \
    {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
     'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
     "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
     'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
     'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
     'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
     'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
     'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
     'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
     's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
     'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
     "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
     'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
     "wouldn't"}


def escape(input):
    return input.translate({
        ord('('): None,
        ord(')'): None,
        ord('\''): None,
        ord('\"'): None,
        ord('.'): ' ',
        ord(':'): ' ',
        ord('\t'): ' ',
        ord('/'): ' ',
        ord('&'): ' ',
        ord(','): ' ',
        ord('-'): ' ',
        ord('?'): ' ',
        ord('+'): ' ',
        ord(';'): ' ',
        ord('`'): None,
        ord('$'): None,
        ord('<'): ' ',
        ord('>'): ' ',
        ord('%'): ' ',
        ord('@'): ' ',
        ord('\\'): ' ',
        ord('*'): ' ',
        ord('!'): ' ',
        ord('['): '',
        ord(']'): '',
        ord('#'): '',
        ord('='): ' ',
        ord('^'): '',
        ord('_'): ' ',
        ord('{'): ' ',
        ord('}'): ' ',
        ord('~'): ' ',
        ord('|'): ' ',
    })


def clean(sentence):
    return ''.join([c if c in char_set else ' ' for c in sentence])


def tokenize(sentence, stem=True, remove_stop=True):
    stemmer = krovetzstemmer.Stemmer()
    clean_sentence = clean(escape(sentence)).lower().strip()
    words = clean_sentence.split()

    if remove_stop:
        words = [w for w in words if w not in stopword_set]

    if stem:
        words = [stemmer.stem(w) for w in words]

    words = list(filter(lambda x: len(x) > 0, [w.strip() for w in words]))

    return words