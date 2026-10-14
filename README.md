# PDF Question Answering CLI (RAG)

**Name:** Yuvraj Singh  **Roll No.:** 2024Btech150
**Course:** LLMs (1 credit)
**Assignment:** Assignment 3 - Convert the PDF search tool into a Retrieval-Augmented Generation (RAG) system

## Problem Statement

> Continue building the command line tool and convert it into a RAG system. Instead of printing the most relevant chunk and similarity score, provide the (query, chunks) to a LLM and output the augmented prompt along with the generated response.

## What is RAG?

A language model only knows what it saw in training, so it cannot answer questions about your PDF and may invent an answer. RAG fixes this in three steps:

1. **Retrieve** the passages of the document that are most relevant to the question.
2. **Augment** the prompt by placing those passages next to the question.
3. **Generate** an answer with the LLM, instructed to use only those passages.

This tool prints the augmented prompt (step 2) and the generated response (step 3), as the assignment asks.

## Pipeline

```text
PDF -> page text -> sentence-aware chunks -> TF-IDF retrieval (top-k)
    -> augmented prompt (rules + chunks + question) -> LLM -> answer + citation check
```

1. **Extract:** PyMuPDF reads the selectable text and keeps each PDF page number.
2. **Chunk:** Each page is split at sentence boundaries into chunks of about 150 words. Neighboring chunks share about 30 words (whole sentences) so ideas that cross a boundary are not lost. Chunks never cross pages, so every citation points to one page.
3. **Retrieve:** Scikit-learn builds TF-IDF vectors (single words and two-word phrases) and ranks chunks by cosine similarity to the question. A small stemmer treats *predict*, *predicting* and *predictions* as the same word.
4. **Augment:** The top chunks are numbered with their page and combined with the question and answering rules into one prompt.
5. **Generate:** The OpenAI Python SDK sends a system instruction and the augmented prompt to Groq or Gemini through their OpenAI-compatible endpoints.
6. **Verify:** The tool checks that every page the model cites was actually in the retrieved context, and warns if not.

## Improvements over the first version

| Problem observed | Fix |
|---|---|
| Chunks were cut mid-sentence, e.g. ending in "...to solve for a (y-intercept) and" | Sentence-aware chunking with sentence-level overlap |
| Tiny chunks (a quiz question, a one-line sentence) ranked high because cosine similarity favors short text | Pages and page-end leftovers under 30 words are skipped or merged into the previous chunk |
| "predictions" did not match "predicting" or "prediction" | Light stemming in the TF-IDF analyzer |
| The model added facts that were not in the retrieved text | Stricter prompt, low temperature (0.1), and a "do not use outside knowledge" rule |
| LaTeX symbols appeared in terminal output | The prompt asks for plain-text formulas |
| Hard to see why a chunk was retrieved | `--show-scores` prints page and similarity per chunk |
| Could not test the prompt without spending API calls | `--dry-run` prints the augmented prompt only |
| Answers could be cut off silently | Warning when the model hits the token limit |

## Setup

Python 3.9 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Configure an LLM provider

The default provider is Groq. Gemini is also supported. Create an API key with the provider you want and set it in your terminal. Never put API keys in the source code or commit them to GitHub.

```bash
export GROQ_API_KEY="your-groq-key"        # default provider
export GEMINI_API_KEY="your-gemini-key"    # only for --provider gemini
```

On Windows PowerShell, use `$env:GROQ_API_KEY="your-groq-key"`.

| Provider | Default model |
|---|---|
| `groq` | `openai/gpt-oss-20b` |
| `gemini` | `gemini-3.5-flash-lite` |

API pricing, free-tier limits and model availability can change. Check the provider's console before use.

## Usage

```bash
python pdf_search.py PDF_FILE "your question" [options]
```

```bash
# Default (Groq)
python pdf_search.py textbook.pdf "How does linear regression make predictions?"

# Gemini
python pdf_search.py textbook.pdf "How does linear regression make predictions?" --provider gemini

# Show which chunks were retrieved and their scores
python pdf_search.py textbook.pdf "What is a decision tree?" --show-scores

# Inspect the augmented prompt without calling the LLM
python pdf_search.py textbook.pdf "What is a decision tree?" --dry-run

# Save the full run (prompt and answer) to a file
python pdf_search.py textbook.pdf "Explain random forests" --save run1.md
```

### Options

| Option | Default | Description |
|---|---:|---|
| `pdf` | required | Path to a selectable-text PDF |
| `query` | required | Question to answer using the PDF |
| `--provider` | `groq` | `groq` or `gemini` |
| `--model` | provider default | Override the model ID |
| `--top-k` | `5` | Chunks passed to the LLM |
| `--chunk-words` | `150` | Target words per chunk (chunks end at sentence boundaries) |
| `--overlap-words` | `30` | Words repeated between neighboring chunks |
| `--min-words` | `30` | Skip pages and page-end leftovers shorter than this |
| `--temperature` | `0.1` | Low values give more factual answers |
| `--max-tokens` | `800` | Maximum answer length |
| `--show-scores` | off | Print page and similarity score of each retrieved chunk |
| `--dry-run` | off | Print the augmented prompt and stop (no API key needed) |
| `--save FILE` | off | Save the run as a Markdown file |

## Example Run

```text
[Paste the full terminal output of your final run here: the augmented prompt and the generated response]
```

## Design Choices

- **TF-IDF retrieval:** It is fast, runs offline, and every retrieved chunk can be explained by the words it shares with the question. It was also the retriever built in Assignment 1.
- **Page-bounded chunks:** Citations always refer to a single PDF page.
- **Context before question:** The prompt places the passages first and the question last, which helps the model focus on the question.
- **Citation check:** Page numbers cited by the model are compared with the retrieved pages, so invented citations are caught.
- **Keys in environment variables:** Keys never appear in the code or the repository.

## Limitations

- TF-IDF matches words, not meaning. A question phrased with different vocabulary from the textbook may retrieve weak chunks. Embedding-based retrieval would handle this better.
- The answer can only be as complete as the retrieved chunks. Raising `--top-k` adds context but also sends more text to the provider.
- The stemmer is simple and will occasionally merge or miss word forms.
- The tool needs internet access and a valid API key to generate answers, and it does not perform OCR on scanned PDFs.
- LLM output is generated text. Check it against the retrieved context printed above the answer.

## Files

```text
assignment-1-pdf-search/
├── pdf_search.py     # chunking, TF-IDF retrieval, prompt construction, LLM call
├── requirements.txt  # pymupdf, scikit-learn, openai
├── README.md         # setup, usage and RAG explanation
└── textbook.pdf      # local test PDF; excluded from Git
```