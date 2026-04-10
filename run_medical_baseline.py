"""
Exercise 3.1 - Baseline Comparison on Medical QA Data
"""
from pathlib import Path
from src import ChunkingStrategyComparator

# Medical QA documents
docs = [
    "data/medical_chatdoctor.md",
    "data/medical_vimed.md"
]

comparator = ChunkingStrategyComparator()

print("=" * 80)
print("Exercise 3.1 — Baseline Comparison (Medical QA Domain)")
print("=" * 80)
print()

results_summary = []

for doc_path in docs:
    path = Path(doc_path)
    if not path.exists():
        continue
    
    text = path.read_text(encoding='utf-8')
    doc_name = path.stem
    
    print(f"Document: {doc_name}")
    print(f"Length: {len(text):,} characters")
    print("-" * 80)
    
    # Run comparison with chunk_size=300 (optimal for QA pairs)
    results = comparator.compare(text, chunk_size=300)
    
    for strategy_name, stats in results.items():
        print(f"\n{strategy_name.upper()}:")
        print(f"  Chunk count: {stats['count']}")
        print(f"  Avg length: {stats['avg_length']:.1f} chars")
        
        # Evaluate context preservation
        if 250 <= stats['avg_length'] <= 350:
            context_quality = "Excellent"
        elif 200 <= stats['avg_length'] <= 400:
            context_quality = "Good"
        else:
            context_quality = "Fair"
        
        results_summary.append({
            'doc': doc_name,
            'strategy': strategy_name,
            'count': stats['count'],
            'avg_len': stats['avg_length'],
            'quality': context_quality
        })
    
    print("\n" + "=" * 80)
    print()

print("\n📊 Summary Table for Report (Section 3):")
print()
print("| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |")
print("|-----------|----------|-------------|------------|-------------------|")

for r in results_summary:
    print(f"| {r['doc']} | {r['strategy']} | {r['count']} | {r['avg_len']:.0f} | {r['quality']} |")

print()
print("=" * 80)
print("Domain Analysis: Medical QA")
print("=" * 80)
print("- QA pairs have natural boundaries (Question → Answer)")
print("- SentenceChunker likely works well for preserving Q&A structure")
print("- FixedSizeChunker provides consistent embedding sizes")
print("- RecursiveChunker may split Q&A pairs awkwardly")
