"""
Exercise 3.1 - Baseline Comparison
Run ChunkingStrategyComparator on sample documents
"""

from pathlib import Path
from src import ChunkingStrategyComparator

# Select 2-3 documents for baseline comparison
docs = [
    "data/python_intro.txt",
    "data/rag_system_design.md",
    "data/vector_store_notes.md"
]

comparator = ChunkingStrategyComparator()

print("=" * 80)
print("Exercise 3.1 — Baseline Comparison")
print("=" * 80)
print()

for doc_path in docs:
    path = Path(doc_path)
    if not path.exists():
        print(f"⚠️  File not found: {doc_path}")
        continue
    
    text = path.read_text(encoding='utf-8')
    doc_name = path.stem
    
    print(f"Document: {doc_name}")
    print(f"Length: {len(text)} characters")
    print("-" * 80)
    
    # Run comparison with chunk_size=200
    results = comparator.compare(text, chunk_size=200)
    
    for strategy_name, stats in results.items():
        print(f"\n{strategy_name.upper()}:")
        print(f"  Chunk count: {stats['count']}")
        print(f"  Avg length: {stats['avg_length']:.1f} chars")
        
        # Show first chunk as example
        if stats['chunks']:
            first_chunk = stats['chunks'][0]
            preview = first_chunk[:100].replace('\n', ' ')
            print(f"  First chunk preview: {preview}...")
    
    print("\n" + "=" * 80)
    print()

print("\nSummary Table for Report:")
print()
print("| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |")
print("|-----------|----------|-------------|------------|-------------------|")

for doc_path in docs:
    path = Path(doc_path)
    if not path.exists():
        continue
    
    text = path.read_text(encoding='utf-8')
    doc_name = path.stem
    results = comparator.compare(text, chunk_size=200)
    
    for strategy_name, stats in results.items():
        # Simple heuristic: if avg_length is close to target, context is likely preserved
        context_quality = "Good" if 150 <= stats['avg_length'] <= 250 else "Fair"
        print(f"| {doc_name} | {strategy_name} | {stats['count']} | {stats['avg_length']:.0f} | {context_quality} |")
