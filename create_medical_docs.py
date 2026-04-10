"""Create 5 medical QA documents from parquet files"""
import pandas as pd
from pathlib import Path

print("Creating medical QA documents...")

# Create output directory
output_dir = Path('data')

# 1. ChatDoctor - 10 QA pairs
print("1. ChatDoctor_Dermatology...")
df1 = pd.read_parquet('data/ChatDoctor_Dermatology_QA.parquet')
sample1 = df1.head(10)

with open(output_dir / 'medical_chatdoctor.md', 'w', encoding='utf-8') as f:
    f.write("# ChatDoctor Dermatology QA\n\n")
    for i, row in sample1.iterrows():
        f.write(f"## Question {i+1}\n{row['question']}\n\n")
        f.write(f"## Answer {i+1}\n{row['answer']}\n\n---\n\n")

print(f"   Created: medical_chatdoctor.md ({Path(output_dir / 'medical_chatdoctor.md').stat().st_size} bytes)")

# 2. ViMedAQA - 10 QA pairs (Vietnamese)
print("2. ViMedAQA...")
df2 = pd.read_parquet('data/ViMedAQA.parquet')
sample2 = df2.head(10)

with open(output_dir / 'medical_vimed.md', 'w', encoding='utf-8') as f:
    f.write("# ViMedAQA - Vietnamese Medical QA\n\n")
    for i, row in sample2.iterrows():
        f.write(f"## Câu hỏi {i+1}\n{row['question']}\n\n")
        f.write(f"## Trả lời {i+1}\n{row['answer']}\n\n---\n\n")

print(f"   Created: medical_vimed.md ({Path(output_dir / 'medical_vimed.md').stat().st_size} bytes)")

print("\nDone! Created 2 medical QA documents.")
print("\nMetadata schema:")
print("- language: en/vi")
print("- category: dermatology/general_medical")
print("- source: chatdoctor/vimed")
print("- type: qa_pair")
