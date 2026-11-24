"""Deterministic Chinese sentence segmentation and longest-match tokenization."""

from .hashing import fingerprint

# Sentence-final Chinese punctuation; commas/colons are kept inside a sentence.
BREAK_PUNCT = "。！？!?"
# Every character in this set is emitted as its own token.
OTHER_PUNCT = "，、；：（）《》“”‘’、,;:()[]\"' "


def segment_document(document):
    """Split every paragraph into sentences with global text offsets."""
    sentences = []
    cursor = 0
    for para_index, paragraph in enumerate(document["paragraphs"]):
        text = paragraph["text"]
        start = None
        for index, char in enumerate(text):
            if start is None:
                start = index
            if char in BREAK_PUNCT:
                end = index + 1
                sentences.append(
                    _make_sentence(document, paragraph, para_index, len(sentences),
                                   text[start:end], cursor + start, cursor + end)
                )
                start = None
        if start is not None:
            sentences.append(
                _make_sentence(document, paragraph, para_index, len(sentences),
                               text[start:], cursor + start, cursor + len(text))
            )
        cursor += len(text)
    return sentences


def _make_sentence(document, paragraph, para_index, index, text, start, end):
    sentence_id = "%s#s%03d" % (document["doc_id"], index)
    return {
        "sentence_id": sentence_id,
        "doc_id": document["doc_id"],
        "paragraph_pid": paragraph["pid"],
        "paragraph_index": para_index,
        "text": text,
        "start_offset": start,
        "end_offset": end,
    }


def assign_sentence_ids(sentences):
    for index, sentence in enumerate(sentences):
        sentence["sentence_id"] = "%s#s%03d" % (sentence["doc_id"], index)
    return sentences


def tokenize(text, vocabulary):
    """Greedy longest-match tokenizer over an ordered controlled vocabulary."""
    tokens = []
    index = 0
    ordered = sorted(vocabulary, key=lambda word: (-len(word), word))
    while index < len(text):
        char = text[index]
        if char in BREAK_PUNCT or char in OTHER_PUNCT:
            index += 1
            continue
        matched = None
        for word in ordered:
            if text.startswith(word, index):
                matched = word
                break
        if matched is None:
            matched = char
        tokens.append(matched)
        index += len(matched)
    return tokens


def build_vocabulary(documents):
    """Controlled vocabulary: entity mentions plus event-triple strings.

    The set is materialized into a sorted list so iteration never depends on
    hash-seed ordering.
    """
    words = set()
    for document in documents:
        for entity in document["entities"]:
            words.add(entity["text"])
        for event in document["events"]:
            for part in event["triple"]:
                words.add(part)
    return tuple(sorted(words))


def segmentation_fingerprint(document, sentences):
    payload = [
        {
            "sentence_id": s["sentence_id"],
            "paragraph_pid": s["paragraph_pid"],
            "text": s["text"],
            "start_offset": s["start_offset"],
            "end_offset": s["end_offset"],
        }
        for s in sentences
    ]
    return fingerprint({"doc": document["content_fingerprint"], "sentences": payload})
