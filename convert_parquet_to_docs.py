"""
Convert parquet files to .md documents for RAG system
Select diverse samples from each dataset
"""
import pandas as pd
from pathlib import Path

# Read all parquet files
datasets = {
    'ChatDoctor_Dermatology': 'data/ChatDoctor_Dermatology_QA.parquet',
    'Dermatology_FineTune': 'data/Dermatology-Question-Answer-Dataset-For-Fine-Tuning.parquet',
    'MedQuAD': 'data/MedQuAD.parquet',
    'MM_Skin': 'data/MM-Skin.parquet',
    'ViMedAQA': 'data/ViMedAQA.parquet'
}

output_dir = Path('data/medical_qa_docs')
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("Converting Parquet to Documents")
print("=" * 80)
print()

for name, filepath in datasets.items():
    try:
        df = pd.read_parquet(filepath)
        print(f"Processing {name}: {len(df)} rows")
        
        # Sample 50 QA pairs from each dataset (or all if less than 50)
        sample_size = min(50, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        # Create markdown document
        output_file = output_dir / f"{name}.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# {name.replace('_', ' ')} - Medical QA Dataset\n\n")
            f.write(f"Source: {filepath}\n")
            f.write(f"Total QA pairs: {len(df)}\n")
            f.write(f"Sample size: {sample_size}\n\n")
            f.write("---\n\n")
            
            for idx, row in df_sample.iterrows():
                f.write(f"## Q{idx + 1}\n\n")
                f.write(f"**Question:** {row['question']}\n\n")
                f.write(f"**Answer:** {row['answer']}\n\n")
                f.write("---\n\n")
        
        # Get file size
        file_size = output_file.stat().st_size
        print(f"  Created: {output_file.name} ({file_size:,} bytes)")
        
    except Exception as e:
        print(f"  Error processing {name}: {e}")

print()
print("=" * 80)
print("Summary")
print("=" * 80)
print(f"Output directory: {output_dir}")
print(f"Files created: {len(list(output_dir.glob('*.md')))}")
print()

# List created files with metadata
print("| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |")
print("|---|--------------|-------|----------|-----------------|")

for i, md_file in enumerate(sorted(output_dir.glob('*.md')), 1):
    content = md_file.read_text(encoding='utf-8')
    char_count = len(content)
    
    # Determine language and category
    if 'ViMed' in md_file.stem:
        language = 'vi'
        category = 'vietnamese_medical'
    else:
        language = 'en'
        category = 'dermatology' if 'Dermatology' in md_file.stem or 'Skin' in md_file.stem else 'general_medical'
    
    metadata = f"language={language}, category={category}, source={md_file.stem}"
    
    print(f"| {i} | {md_file.stem} | {md_file.name} | {char_count:,} | {metadata} |")
