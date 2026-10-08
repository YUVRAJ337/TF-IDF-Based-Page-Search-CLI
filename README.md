# Assignment 1: PDF Page Search (Command-Line Tool)

**Name:** YUVRAJ_SINGH  **Roll No.:** 2024BTech150
**Course:** LLMs (1 credit)

## Problem Statement

> Select a PDF with a large number of pages (~100) with selectable text. Create a command-line tool that outputs the page number of the page (and the relevant text) that is most similar to a given query.

## What This Tool Does

`pdf_search.py` takes a PDF and a text query, then prints the pages most similar to the query, with each page's number, a similarity score, and a snippet of the relevant text.

**Test document:** MAchine learning for absolute beginners, 179 pages, selectable text.

## How It Works

```
PDF ──► text per page ──► TF-IDF vectors ──► cosine similarity vs. query ──► ranked pages
        (PyMuPDF)         (scikit-learn)      (scikit-learn)                  + snippet
```

1. **Extract:** PyMuPDF reads each page's text and collapses extra whitespace. Page numbers start at 1 and match the PDF viewer.
2. **Vectorize:** `TfidfVectorizer` turns every page into a vector of word weights.
   - `stop_words="english"` drops common words such as *the* and *is*.
   - `sublinear_tf=True` uses `1 + log(tf)`, so a word repeated 50 times does not dominate.
   - `ngram_range=(1, 2)` counts two-word phrases, so *gradient descent* is a single feature.
3. **Score:** The query is vectorized with the same vocabulary, and cosine similarity is computed against every page:

   ```
   cosine(q, p) = (q · p) / (‖q‖ · ‖p‖)
   ```

4. **Rank and display:** Pages are sorted by score, and pages with a score of 0 are skipped. For each result, the snippet is the window of text containing the most distinct query words, not just the start of the page.

## Setup

```bash
pip install pymupdf scikit-learn
```

Python 3.9+ is recommended (tested on Python 3.13, macOS).

## Usage

```bash
python pdf_search.py <pdf> "<query>" [--top-k N] [--snippet CHARS]
```

| Argument | Default | Meaning |
|---|---|---|
| `pdf` | required | Path to a PDF with selectable text |
| `query` | required | Search query |
| `--top-k` | 3 | Number of matching pages to show |
| `--snippet` | 500 | Approximate characters of text shown per page |

## Example Runs

All runs below use `textbook.pdf` (179 pages).

### 1. Phrase query

```
$ python pdf_search.py textbook.pdf "LINEAR REGRESSION" --top-k 3
Loaded 179 pages from textbook.pdf

Query: "LINEAR REGRESSION"
============================================================
#1  Page 53  (similarity: 0.2186)
7 LINEAR REGRESSION As the "Hello World" of supervised learning algorithms, regression
analysis is a simple technique for predicting an unknown variable using the results you do
know. The first regression technique we'll examine is linear regression, which generates a
straight line to describe linear relationships. ...
------------------------------------------------------------
#2  Page 57  (similarity: 0.2023)
Figure 17: Using the slope/hyperplane to make a prediction While linear regression isn't a
fail-proof method for predicting trends, ... Linear Regression Formula The formula [17] for
linear regression is y = bx + a. ...
------------------------------------------------------------
#3  Page 60  (similarity: 0.1846)
Having summarized simple linear regression using a single independent variable, we can now
look at multiple linear regression. ...
------------------------------------------------------------
```

Page 53 is the first page of the linear regression chapter, so it is the correct top match. The query is matched case-insensitively.

### 2. Specific term

```
$ python pdf_search.py textbook.pdf "gradient descent" --top-k 3
Loaded 179 pages from textbook.pdf

Query: "gradient descent"
============================================================
#1  Page 35  (similarity: 0.0882)
...ries. While Scikit-learn offers a range of popular shallow algorithms, TensorFlow is
the machine learning library of choice for deep learning/neural networks. It supports
numerous advanced techniques including automatic calculus for back-propagation/gradient
descent. ...
------------------------------------------------------------
#2  Page 131  (similarity: 0.0590)
...HAPTER QUIZ ... 2) To train a quick benchmark model, gradient boosting is faster to
train than random forests. ...
------------------------------------------------------------
#3  Page 132  (similarity: 0.0419)
ANSWERS ... (Gradient boosting runs sequentially, making it slower to train. ...
------------------------------------------------------------
```

This textbook mentions gradient descent only in passing (page 35). The tool finds that page, and the lower scores show the book has no dedicated section on it. Pages 131–132 match on the word *gradient* alone (gradient boosting).

### 3. Natural-language question

```
$ python pdf_search.py textbook.pdf "how does a decision tree work?"
Loaded 179 pages from textbook.pdf

Query: "how does a decision tree work?"
============================================================
#1  Page 132  (similarity: 0.1230)
ANSWERS 1) All variables except for body_mass_g (Tree-based techniques work well with both
discrete and continuous variables ...) 3) A, Decision trees
------------------------------------------------------------
#2  Page 125  (similarity: 0.1169)
...icks a variable that best partitions the data into homogenous groups. ... the greedy
algorithm is oblivious to the future repercussions of its short-term actions. ...
------------------------------------------------------------
#3  Page 118  (similarity: 0.1143)
Figure 55: Example of a regression tree Figure 56: Example of a classification tree Part
of the appeal of decision trees is they can be displayed graphically and they are easy to
explain to non-experts. ...
------------------------------------------------------------
```

All three pages are from the decision-tree chapter. Page 132, a short quiz-answers page, ranks first (see Limitations).

## Design Choices

- **TF-IDF over neural embeddings:** The tool runs fully offline, needs no API key, and is fast. A 179-page PDF is searched in about a second. Because it matches shared words, every result can be explained by the words in the snippet.
- **Bigrams:** Phrases like *linear regression* and *decision tree* are matched as units, which improves ranking over single words.
- **Query-centered snippets:** The snippet is chosen around the query words, so the output shows the relevant text instead of the first lines of the page.
- **Input checks:** The tool exits with a clear message for a missing file, a non-PDF file, a PDF with no selectable text (scanned images), or a query with no words found in the PDF.

## Limitations

- **Exact words only:** TF-IDF does not understand meaning. A paraphrase such as "how do networks learn from mistakes?" will not match a page that says "back-propagation" unless the words overlap. Embedding-based search would handle this better.
- **Short pages can win:** In run 3, a short quiz-answers page (132) outranked the explanatory pages because cosine similarity favors pages where query words make up a large share of the text. Filtering out very short pages would help.
- **Low absolute scores:** Scores such as 0.09 are normal for TF-IDF with bigrams over many pages. Only the ranking matters, and scores should not be read as percentages.
- **Scanned PDFs are not supported** because there is no OCR step.

## Files

```
assignment-1-pdf-search/
├── pdf_search.py   # the tool
└── README.md
```

The test PDF is not included because of its size and copyright.