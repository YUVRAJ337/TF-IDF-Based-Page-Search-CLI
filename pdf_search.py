#!/usr/bin/env python3
"""Find the PDF pages whose text is most similar to a query using TF-IDF."""

import argparse
import re
import sys
import textwrap

import pymupdf
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


def extract_pages(pdf_path):
    """Return (PDF page number, extracted text) pairs. Page numbers start at 1."""
    pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            # Collapse line breaks and repeated whitespace for easier searching.
            text = " ".join(page.get_text().split())
            pages.append((page_number, text))

    return pages


def make_snippet(text, query, length):
    """Choose a snippet around query words that appear on this page."""
    query_words = {
        word.lower()
        for word in re.findall(r"\b\w+\b", query)
        if word.lower() not in ENGLISH_STOP_WORDS
    }

    # Find the positions of words on this page that also appear in the query.
    words = list(re.finditer(r"\b\w+\b", text))
    matching_words = [
        word for word in words if word.group().lower() in query_words
    ]

    if not matching_words:
        return text[:length] + ("..." if len(text) > length else "")

    # Try a snippet around each matching word. Keep the one containing the
    # greatest number of distinct query words.
    best_start = 0
    best_count = -1

    for match in matching_words:
        start = max(0, match.start() - length // 2)
        end = min(len(text), start + length)
        snippet_text = text[start:end].lower()

        count = sum(
            1 for word in query_words
            if re.search(r"\b" + re.escape(word) + r"\b", snippet_text)
        )

        if count > best_count:
            best_start = start
            best_count = count

    end = min(len(text), best_start + length)
    snippet = text[best_start:end]

    if best_start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet += "..."

    return snippet


def search_pages(pages, query):
    """Return a similarity score for each page."""
    page_texts = [text for _, text in pages]

    # TF-IDF turns the pages into word and two-word-phrase vectors.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
    )

    try:
        page_vectors = vectorizer.fit_transform(page_texts)
        query_vector = vectorizer.transform([query])
    except ValueError as error:
        if "empty vocabulary" in str(error).lower():
            sys.exit("Could not find searchable words in the PDF or query.")
        raise

    if query_vector.nnz == 0:
        sys.exit(
            "None of the query's searchable words appear in the PDF. "
            "Try different or simpler query words."
        )

    return cosine_similarity(query_vector, page_vectors)[0]


def main():
    parser = argparse.ArgumentParser(
        description="Find the PDF pages most similar to a text query using TF-IDF."
    )
    parser.add_argument("pdf", help="Path to a PDF with selectable text")
    parser.add_argument("query", help='Search query, for example "gradient descent"')
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of matching pages to display (default: 3)",
    )
    parser.add_argument(
        "--snippet",
        type=int,
        default=500,
        help="Approximate number of characters to show per page (default: 500)",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.snippet < 1:
        parser.error("--snippet must be at least 1")

    try:
        pages = extract_pages(args.pdf)
    except FileNotFoundError:
        sys.exit(f"PDF not found: {args.pdf}")
    except pymupdf.FileDataError:
        sys.exit(f"Could not open this file as a PDF: {args.pdf}")

    print(f"Loaded {len(pages)} pages from {args.pdf}", file=sys.stderr)

    if not pages or not any(text for _, text in pages):
        sys.exit("No selectable text found. This PDF may contain scanned images.")

    scores = search_pages(pages, args.query)
    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    print(f'\nQuery: "{args.query}"')
    print("=" * 60)

    results_printed = 0

    for index in ranked_indexes:
        score = float(scores[index])

        # Ignore pages that have no matching query terms.
        if score <= 0:
            continue

        page_number, text = pages[index]
        snippet = make_snippet(text, args.query, args.snippet)

        results_printed += 1
        print(f"#{results_printed}  Page {page_number}  (similarity: {score:.4f})")
        print(textwrap.fill(snippet, width=90))
        print("-" * 60)

        if results_printed >= args.top_k:
            break

    if results_printed == 0:
        print("No pages matched the query.")


if __name__ == "__main__":
    main()