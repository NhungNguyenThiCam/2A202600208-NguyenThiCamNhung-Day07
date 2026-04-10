"""
Exercise 3.3 - Cosine Similarity Predictions
Test compute_similarity() on 5 pairs of sentences
"""

from src import compute_similarity, _mock_embed

# Define 5 pairs of sentences
pairs = [
    {
        "id": 1,
        "sentence_a": "Python is a popular programming language for machine learning.",
        "sentence_b": "Machine learning developers often use Python as their primary language.",
        "prediction": "high",
        "reason": "Both sentences discuss Python and machine learning with overlapping vocabulary"
    },
    {
        "id": 2,
        "sentence_a": "The cat sat on the mat.",
        "sentence_b": "A feline rested on the rug.",
        "prediction": "high",
        "reason": "Same meaning, different words (synonyms: cat=feline, sat=rested, mat=rug)"
    },
    {
        "id": 3,
        "sentence_a": "I love eating pizza with extra cheese.",
        "sentence_b": "The weather forecast predicts rain tomorrow.",
        "prediction": "low",
        "reason": "Completely different topics (food vs weather), no vocabulary overlap"
    },
    {
        "id": 4,
        "sentence_a": "Vector databases store embeddings for similarity search.",
        "sentence_b": "Embeddings are vector representations used in similarity search.",
        "prediction": "high",
        "reason": "Both about embeddings and similarity search, high vocabulary overlap"
    },
    {
        "id": 5,
        "sentence_a": "The quick brown fox jumps over the lazy dog.",
        "sentence_b": "Artificial intelligence is transforming modern technology.",
        "prediction": "low",
        "reason": "Unrelated topics (animal action vs AI technology), no semantic connection"
    }
]

print("=" * 80)
print("Exercise 3.3 — Cosine Similarity Predictions")
print("=" * 80)
print()

results = []

for pair in pairs:
    # Get embeddings
    emb_a = _mock_embed(pair["sentence_a"])
    emb_b = _mock_embed(pair["sentence_b"])
    
    # Compute similarity
    score = compute_similarity(emb_a, emb_b)
    
    # Determine if prediction was correct
    actual = "high" if score > 0.5 else "low"
    correct = "✓" if actual == pair["prediction"] else "✗"
    
    results.append({
        **pair,
        "actual_score": score,
        "actual": actual,
        "correct": correct
    })
    
    print(f"Pair {pair['id']}: {correct}")
    print(f"  Sentence A: {pair['sentence_a']}")
    print(f"  Sentence B: {pair['sentence_b']}")
    print(f"  Prediction: {pair['prediction']}")
    print(f"  Actual Score: {score:.4f} ({actual})")
    print(f"  Reason: {pair['reason']}")
    print()

print("=" * 80)
print("Summary")
print("=" * 80)

correct_count = sum(1 for r in results if r["correct"] == "✓")
print(f"Correct predictions: {correct_count} / {len(results)}")
print()

# Find most surprising result
print("Most surprising result:")
for r in results:
    if r["correct"] == "✗":
        print(f"  Pair {r['id']}: Predicted {r['prediction']}, but got {r['actual']} (score={r['actual_score']:.4f})")
        print(f"  This suggests: Mock embeddings may not capture semantic similarity well,")
        print(f"  or the threshold (0.5) needs adjustment for this embedding model.")
        break
else:
    # If all correct, find the one closest to threshold
    closest = min(results, key=lambda r: abs(r["actual_score"] - 0.5))
    print(f"  Pair {closest['id']}: Score {closest['actual_score']:.4f} was close to threshold (0.5)")
    print(f"  This shows the boundary between high/low similarity is not always clear-cut.")

print()
print("=" * 80)
print("Table for Report")
print("=" * 80)
print()
print("| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |")
print("|------|-----------|-----------|---------|--------------|-------|")
for r in results:
    print(f"| {r['id']} | {r['sentence_a'][:40]}... | {r['sentence_b'][:40]}... | {r['prediction']} | {r['actual_score']:.4f} | {r['correct']} |")
