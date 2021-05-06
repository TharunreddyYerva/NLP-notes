# NLP-notes

# RegEx
RegEx
While dealing with unstructured data RegularExpression or RegEx are mainly used to find and replace patterns in string or files.

Common uses of RegEx are:

1.Search a sttring

2.Find a string

3.Replace a part of string

hat are regular expressoins
. pattern of special characters having an textual meaning(ex: "\id":'numbers')

. also called wild card expression for matching,searching and parsing strings

. used for writing rule-based information mining systems

Applications
. segmentation of words from sentences

. segmentation of sentences from paragraphs also called tokenization

. Text cleaning- removal of unnecessery noise

Types of regular expressions
ex: abc---- letters

123-----numbers

\d------any digit

[abc]---only a,b or c

[0-9]---numbers from 0 to 9
function
1.match: finds the first occurance of pattern in the string

2.search:locates the pattern in the string

3.findall: find all occurance of pattern in the string

4.sub: search and replace

5.split: split the given text by using given expression

implementation regular expression in python
import re
​
string = "tiger is the national animal of the india"
​
pattern = 'tiger'
​
# above we have string task is find the pattern in the above string
​
result = re.match(pattern, string).group(0)
​
# match function search the string it is present in the first place or not. if it is present give tiger as result
​
​
result
'tiger'
string1 = 'the national animal of india is tiger'
​
pattern = 'tiger'
​
res = re.search(pattern, string1).group(0)
# re.search function works similar to re.match and it search through entire string
res
'tiger'
srting = "tiger is a national animal and hacky is a national game of india, national "
pattern = 'national'
mo = re.search(pattern,string)
print(mo)
<re.Match object; span=(13, 21), match='national'>
mo.group(0)
'national'
re.finditer(pattern, string)
<callable_iterator at 0x1673ee498e0>
string = 'hero was born on 12-12-1992 and he was admited to school 25-12-2005'
pattern = '\d{2}-\d{2}-\d{4}'# to find years present in string
e = re.search(pattern, string)
e
<re.Match object; span=(17, 27), match='12-12-1992'>
# re.sub used to replace string
re.sub(pattern, 'monday', string)
'hero was born on monday and he was admited to school monday'
in order to work with text data it is necessary to transform the raw text data in to machine understanding data
Text Preprocessing
Corpus, Tokens and N-Grams
Corpus: data set is a collection text documents(pharagraphs, sentences, tokens)

Tokens: smaller units of a text(words, phrases, ngrams)

N-Grams:combination of N words/ characters togrther

ex: i love my phone

unigrams(n=1) i, love, my, phone

bigrams(n=2) i love,love my, my phone

trigrams(n=3) i love my, love my phone
n-grams are very useful in text classification tasks

tokenization
it is a process of splitting text into smaller units(tokens)

. white space tokenization is most used tokenization process. ex: i went to hyderabad

'i', 'went', 'to', 'hyderabad' in this splitting done by white space
Normalization
. normalization is a process of converting token in to a base form(morphin)

.morpheme: base form of words, structure of token is morphin

1.Stemming: elementary rule based process of removal of inflectional forms from token

2.Lammatization: systematic process for reducing token to its lemma

part of speech tags(pos)
. define the syntactic context and role of words in the sentence

. common pos tags: nouns, verbs, Adjectives, Adverbs

. define by their relationship with the adjacent words

. uses

. text cleaning

. feature engineering tasks

. word sense disambiguation
constituency grammer
constituents: words,phases,group of words

Dependency grammer
words of sentence depends on which other words(dependencies)

use cases:

    named entity recognition( NER )

    Quetion anwsering system

    conference resolution

    text summarization

    text classification
CountVectorizer
In order to use textual data for predictive modeling, the text must be parsed to remove certain words – this process is called tokenization. These words need to then be encoded as integers, or floating-point values, for use as inputs in machine learning algorithms. This process is called feature extraction (or vectorization).

Scikit-learn’s CountVectorizer is used to convert a collection of text documents to a vector of term/token counts. It also enables the pre-processing of text data prior to generating the vector representation. This functionality makes it a highly flexible feature representation module for text.

image.png

from sklearn.feature_extraction.text import CountVectorizer
​
# list of text documents
text = ["John is a good boy. John watches basketball"]
​
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
​
print(vectorizer.vocabulary_)
​
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(vector.toarray())
{'john': 4, 'is': 3, 'good': 2, 'boy': 1, 'watches': 5, 'basketball': 0}
(1, 6)
[[1 1 1 1 2 1]]
Tfidf vectorizer
image.png

image.png

image.png

​
​
