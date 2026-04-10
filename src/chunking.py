from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        
        # Split by sentence boundaries: ". ", "! ", "? ", ".\n"
        sentences = re.split(r'(?<=[.!?])\s+|\.\n', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Group sentences into chunks
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_sentences = sentences[i:i + self.max_sentences_per_chunk]
            chunk = " ".join(chunk_sentences)
            chunks.append(chunk)
        
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case: if text fits in chunk_size, return it
        if len(current_text) <= self.chunk_size:
            return [current_text] if current_text else []
        
        # Base case: no more separators, split by character
        if not remaining_separators:
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[i:i + self.chunk_size])
            return chunks
        
        # Try to split by the first separator
        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]
        
        if separator and separator in current_text:
            # Split by this separator
            parts = current_text.split(separator)
            chunks: list[str] = []
            
            for i, part in enumerate(parts):
                # Re-add separator except for last part
                if i < len(parts) - 1:
                    part = part + separator
                
                if len(part) <= self.chunk_size:
                    chunks.append(part)
                else:
                    # Part is too large, recursively split with next separators
                    chunks.extend(self._split(part, next_separators))
            
            return [c for c in chunks if c]
        else:
            # Separator not found, try next separator
            return self._split(current_text, next_separators)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    
    dot_product = _dot(vec_a, vec_b)
    
    # Compute magnitudes
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(x * x for x in vec_b))
    
    # Avoid division by zero
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    
    return dot_product / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        results = {}
        
        # Strategy 1: Fixed Size
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=20)
        fixed_chunks = fixed_chunker.chunk(text)
        results['fixed_size'] = {
            'count': len(fixed_chunks),
            'avg_length': sum(len(c) for c in fixed_chunks) / len(fixed_chunks) if fixed_chunks else 0,
            'chunks': fixed_chunks
        }
        
        # Strategy 2: By Sentences
        # Estimate sentences per chunk based on chunk_size
        avg_sentence_length = 80  # rough estimate
        max_sentences = max(1, chunk_size // avg_sentence_length)
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=max_sentences)
        sentence_chunks = sentence_chunker.chunk(text)
        results['by_sentences'] = {
            'count': len(sentence_chunks),
            'avg_length': sum(len(c) for c in sentence_chunks) / len(sentence_chunks) if sentence_chunks else 0,
            'chunks': sentence_chunks
        }
        
        # Strategy 3: Recursive
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)
        recursive_chunks = recursive_chunker.chunk(text)
        results['recursive'] = {
            'count': len(recursive_chunks),
            'avg_length': sum(len(c) for c in recursive_chunks) / len(recursive_chunks) if recursive_chunks else 0,
            'chunks': recursive_chunks
        }
        
        return results
