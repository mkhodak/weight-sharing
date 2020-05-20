import pdb
from unicodedata import category
from nltk import ngrams, word_tokenize
from nltk.corpus import stopwords
from scipy import sparse as sp
from utils import FLOAT


PUNCTUATION = {'M', 'P', 'S'}
STOP = set.union(*({word, word.capitalize(), ''.join(char for char in word if not category(char)[0] in PUNCTUATION)} for word in stopwords.words('english')))


def memo(func):
    cache = {}
    def wrapper(tagdoc):
        output = cache.get(tagdoc[0])
        if output is None:
            output = list(func(tagdoc[1]))
            cache[tagdoc[0]] = output
        return output
    return wrapper


word_tokenize = memo(word_tokenize)


@memo
def remove_punctuation(doc):
    '''tokenizes stirng by removing punctuation
    Args:
        document: string
    Returns:
        str generator
    '''

    for token in doc.split():
        token = ''.join(char for char in token if not category(char)[0] in PUNCTUATION)
        if token:
            yield token


# from https://github.com/NLPrinceton/text_embedding
@memo
def split_on_punctuation(doc):
  '''tokenizes string by splitting on spaces and punctuation
  Args:
    document: string
  Returns:
    str generator
  '''

  for token in doc.split():
    if len(token) == 1:
      yield token
    else:
      chunk = token[0]
      for char0, char1 in zip(token[:-1], token[1:]):
        if (category(char0)[0] in PUNCTUATION) == (category(char1)[0] in PUNCTUATION):
          chunk += char1
        else:
          yield chunk
          chunk = char1
      if chunk:
        yield chunk


def remove_stopwords(doc):
    '''removes stopwords from tokenized doc
    Args:
        doc: list of strings
    Returns:
        string generator
    '''

    return (token for token in doc if not token in STOP)


def hashed_bongs(docs, hash_func, n_bins, order=1, format='csr'):
    '''constructs hashed bag-of-n-gram features from tokenized docs
    Args:
        docs: list of lists of strings
        hash_func: unsigned hash function
        n_bins: feature dimension
        order: n-gram model order
        format: format of sparse output matrix
    '''

    rows, cols, vals = zip(*((row, hash_func(' '.join(ngram)) % n_bins, 1.0) 
                             for row, doc in enumerate(docs) 
                             for n in range(1, order+1)
                             for ngram in ngrams(doc, n)))
    return sp.coo_matrix((vals, (rows, cols)), shape=(len(docs), n_bins), dtype=FLOAT).asformat(format)
