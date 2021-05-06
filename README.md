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
​important topics
NLP techniques to problems such as text classification, text summarization, question & answering, information retrieval, knowledge extraction, and conversational bots design potentially with both traditional & Deep Learning Techniques, sentiment analysis

important libraries:
Experience on one or more of these skills: BERT, HMM, CRF, LDA, Word2Vec,CountVectorizer,tfidf,Word Embeding Seq2Seq, spaCy, Nltk, Gensim, CoreNLP, NLU, NLG.

what is text classification:
Text classification also known as text tagging or text categorization is the process of categorizing text into organized groups. By using Natural Language Processing (NLP), text classifiers can automatically analyze text and then assign a set of pre-defined tags or categories based on its content.

Unstructured text is everywhere, such as emails, chat conversations, websites, and social media but it’s hard to extract value from this data unless it’s organized in a certain way. Doing so used to be a difficult and expensive process since it required spending time and resources to manually sort the data or creating handcrafted rules that are difficult to maintain. Text classifiers with NLP have proven to be a great alternative to structure textual data in a fast, cost-effective, and scalable way.

Some of the most common examples and use cases for automatic text classification include the following:

1.Sentiment Analysis:
the process of understanding if a given text is talking positively or negatively about a given subject (e.g. for brand monitoring purposes).

2.Topic Detection:
the task of identifying the theme or topic of a piece of text (e.g. know if a product review is about Ease of Use, Customer Support, or Pricing when analyzing customer feedback).

3.Language Detection:
the procedure of detecting the language of a given text (e.g. know if an incoming support ticket is written in English or Spanish for automatically routing tickets to the appropriate team).

what is text summarization in nlp:
Automatic Text Summarization is one of the most challenging and interesting problems in the field of Natural Language Processing (NLP). It is a process of generating a concise and meaningful summary of text from multiple text resources such as books, news articles, blog posts, research papers, emails, and tweets.

The demand for automatic text summarization systems is spiking these days thanks to the availability of large amounts of textual data.

An Introduction to Text Summarization using the TextRank Algorithm (with Python implementation):https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/#:~:text=Text%20Summarization%20is%20one%20of,huge%20impact%20on%20our%20lives.&text=It%20is%20a%20process%20of,papers%2C%20emails%2C%20and%20tweets.

what is information retrieval:
nformation retrieval (IR) may be defined as a software program that deals with the organization, storage, retrieval and evaluation of information from document repositories particularly textual information. The system assists users in finding the information they require but it does not explicitly return the answers of the questions. It informs the existence and location of documents that might consist of the required information. The documents that satisfy user’s requirement are called relevant documents. A perfect IR system will retrieve only relevant documents.

link:https://www.tutorialspoint.com/natural_language_processing/natural_language_processing_information_retrieval.htm

what is knowledge extraction:
Text data contains a lot of information but not all of it will be important to you. We might be looking for names of entities, others would want to extract specific relationships between those entities. Our intentions differ according to our requirements.

link:https://www.analyticsvidhya.com/blog/2020/06/nlp-project-information-extraction/

NLP Library:
BERT (Bidirectional Encoder Representations from Transformers):
BERT is an open source machine learning framework for natural language processing (NLP). BERT is designed to help computers understand the meaning of ambiguous language in text by using surrounding text to establish context. The BERT framework was pre-trained using text from Wikipedia and can be fine-tuned with question and answer datasets.

BERT, which stands for Bidirectional Encoder Representations from Transformers, is based on Transformers, a deep learning model in which every output element is connected to every input element, and the weightings between them are dynamically calculated based upon their connection. (In NLP, this process is called attention.)

Historically, language models could only read text input sequentially -- either left-to-right or right-to-left -- but couldn't do both at the same time. BERT is different because it is designed to read in both directions at once. This capability, enabled by the introduction of Transformers, is known as bidirectionality.

image.png

What is BERT used for?
BERT is currently being used at Google to optimize the interpretation of user search queries. BERT excels at several functions that make this possible, including:

Sequence-to-sequence based language generation tasks such as:

Question answering

Abstract summarization

Sentence prediction

Conversational response generation

Natural language understanding tasks such as:

Polysemy and Coreference (words that sound or look the same but have different meanings) resolution

Word sense disambiguation

Natural language inference

Sentiment classification

Hidden Markov Model (HMM) :
as “a statistical Markov model in which the system being modeled is assumed to be a Markov process with unobserved (hidden) states”.

Hidden Markov Models (HMMs) are a class of probabilistic graphical model that allow us to predict a sequence of unknown (hidden) variables from a set of observed variables. A simple example of an HMM is predicting the weather (hidden variable) based on the type of clothes that someone wears (observed).

image.png

Conditional Random Fields (CRF)
CRF is a discriminant model for sequences data similar to MEMM. It models the dependency between each state and the entire input sequences. Unlike MEMM, CRF overcomes the label bias issue by using global normalizer.

Latent Dirichlet Allocation (LDA)
is one such technique designed to assist in modelling the data consisting of a large corpus of words. There is some terminology that one needs to be familiar with

First, each word in each document is randomly assigned to one of the topics.

Now, it is assumed that all topic assignments except for the current one are correct.

The proportion of words in document say, ‘d’ that are currently assigned to topic ‘t’ is equal to p(topic t | document d) and proportion of assignments topic ‘t’ over all documents that belong to word ‘w’ is equal to p(word w | topic t).

These two proportions are multiplied and assigned a new topic based on that probability.

Word2vec :
is a two-layer neural net that processes text by “vectorizing” words. Its input is a text corpus and its output is a set of vectors: feature vectors that represent words in that corpus. While Word2vec is not a deep neural network, it turns text into a numerical form that deep neural networks can understand.

​
​
​
Data Preprocessing in NLP
What is text preprocessing?
To preprocess your text simply means to bring your text into a form that is predictable and analyzable for your task. A task here is a combination of approach and domain. For example, extracting top keywords with TF-IDF (approach) from Tweets (domain) is an example of a Task.

Types of text preprocessing techniques
There are different ways to preprocess your text. Here are some of the approaches that you should know about and I will try to highlight the importance of each.

Lowercasing
Lowercasing ALL your text data, although commonly overlooked, is one of the simplest and most effective form of text preprocessing. It is applicable to most text mining and NLP problems and can help in cases where your dataset is not very large and significantly helps with consistency of expected output.

image.png

Stemming
Stemming is the process of reducing inflection in words (e.g. troubled, troubles) to their root form (e.g. trouble). The “root” in this case may not be a real root word, but just a canonical form of the original word.Stemming uses a crude heuristic process that chops off the ends of words in the hope of correctly transforming words into its root form. So the words “trouble”, “troubled” and “troubles” might actually be converted to troublinstead of trouble because the ends were just chopped off (ughh, how crude!).

image.png

Lemmatization
Lemmatization on the surface is very similar to stemming, where the goal is to remove inflections and map a word to its root form. The only difference is that, lemmatization tries to do it the proper way. It doesn’t just chop things off, it actually transforms words to the actual root. For example, the word “better” would map to “good”. It may use a dictionary such as WordNet for mappings or some special rule-based approaches. Here is an example of lemmatization in action using a WordNet-based approach.

image.png

Stopword Removal
Stop words are a set of commonly used words in a language. Examples of stop words in English are “a”, “the”, “is”, “are” and etc. The intuition behind using stop words is that, by removing low information words from text, we can focus on the important words instead.top words are commonly applied in search systems, text classification applications, topic modeling, topic extraction and others.In my experience, stop word removal, while effective in search and topic extraction systems, showed to be non-critical in classification systems. However, it does help reduce the number of features in consideration which helps keep your models decently sized.

Normalization
A highly overlooked preprocessing step is text normalization. Text normalization is the process of transforming a text into a canonical (standard) form. For example, the word “gooood” and “gud” can be transformed to “good”, its canonical form. Another example is mapping of near identical words such as “stopwords”, “stop-words” and “stop words” to just “stopwords”.

Text normalization is important for noisy texts such as social media comments, text messages and comments to blog posts where abbreviations, misspellings and use of out-of-vocabulary words (oov) are prevalent. This paper showed that by using a text normalization strategy for Tweets, they were able to improve sentiment classification accuracy by ~4%.

In my experience, text normalization has even been effective for analyzing highly unstructured clinical texts where physicians take notes in non-standard ways. I’ve also found it useful for topic extraction where near synonyms and spelling differences are common (e.g. topic modelling, topic modeling, topic-modeling, topic-modelling).Unfortunately, unlike stemming and lemmatization, there isn’t a standard way to normalize texts. It typically depends on the task. For example, the way you would normalize clinical texts would arguably be different from how you normalize sms text messages.Some common approaches to text normalization include dictionary mappings (easiest), statistical machine translation (SMT) and spelling-correction based approaches. This interesting article compares the use of a dictionary based approach and a SMT approach for normalizing text messages.

image.png

Noise Removal
Noise removal is about removing characters digits and pieces of text that can interfere with your text analysis. Noise removal is one of the most essential text preprocessing steps. It is also highly domain dependent.

For example, in Tweets, noise could be all special characters except hashtags as it signifies concepts that can characterize a Tweet. The problem with noise is that it can produce results that are inconsistent in your downstream tasks. Let’s take the example below:

image.png

Text Enrichment / Augmentation
Text enrichment involves augmenting your original text data with information that you did not previously have. Text enrichment provides more semantics to your original text, thereby improving its predictive power and the depth of analysis you can perform on your data.

In an information retrieval example, expanding a user’s query to improve the matching of keywords is a form of augmentation. A query like text mining could become text document mining analysis. While this doesn’t make sense to a human, it can help fetch documents that are more relevant.You can get really creative with how you enrich your text. You can use part-of-speech tagging to get more granular information about the words in your text.For example, in a document classification problem, the appearance of the word book as a noun could result in a different classification than book as a verb as one is used in the context of reading and the other is used in the context of reserving something. This article talks about how Chinese text classification is improved with a combination of nouns and verbs as input features.With the availability of large amounts texts however, people have started using embeddings to enrich the meaning of words, phrases and sentences for classification, search, summarization and text generation in general. This is especially true in deep learning based NLP approaches where a word level embedding layer is quite common. You can either start with pre-established embeddings or create your own and use it in downstream tasks.

Must Do:
Noise removal Lowercasing (can be task dependent in some cases)

Should Do:
Simple normalization — (e.g. standardize near identical words)

Task Dependent:
Advanced normalization (e.g. addressing out-of-vocabulary words) Stop-word removal Stemming / lemmatization Text enrichment / augmentation

image.png

List of Text Preprocessing Steps
Based on the general outline above, we performed a series of steps under each component.

. Remove HTML tags

. Remove extra whitespaces

. Convert accented characters to ASCII characters

. Expand contractions

. Remove special characters

. Lowercase all texts

. Convert number words to numeric form

. Remove numbers

. Remove stopwords

. Lemmatization
