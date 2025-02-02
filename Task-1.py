import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from heapq import nlargest

nltk.download('punkt')

def summarize_text(text, num_sentences=3):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    word_freq = Counter(words)
    sentence_scores = {}
    
    for sentence in sentences:
        sentence_word_count = word_tokenize(sentence.lower())
        sentence_scores[sentence] = sum(word_freq[word] for word in sentence_word_count if word in word_freq)
    
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = " ".join(summary_sentences)
    
    return summary

text = """Text summarization is the process of creating a short and coherent version of a longer document.
It is useful in applications where quick understanding of large text is required.
There are two types of text summarization: extractive and abstractive.
Extractive summarization selects important sentences from the original text,
while abstractive summarization generates new sentences that convey the main idea."""

summary = summarize_text(text, num_sentences=2)
print("Summary:", summary)
