#!/usr/bin/env python3
"""
Fix MMMU predictions by extracting answers from <answer> tags
"""
import pandas as pd
import re
import sys

def extract_answer_from_tags(text):
    """Extract answer from <answer>...</answer> tags"""
    text = str(text)
    
    # Try to extract from <answer> tags
    match = re.search(r'<answer>(.*?)</answer>', text, re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip()
        return answer
    
    # If no tags found, return original text
    return text

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_mmmu_predictions.py <input_xlsx>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = input_file.replace('.xlsx', '_fixed.xlsx')
    
    print(f"Loading {input_file}...")
    df = pd.read_excel(input_file)
    
    print(f"Extracting answers from {len(df)} predictions...")
    original_lengths = df['prediction'].apply(lambda x: len(str(x)))
    
    # Extract answers
    df['prediction'] = df['prediction'].apply(extract_answer_from_tags)
    
    new_lengths = df['prediction'].apply(lambda x: len(str(x)))
    
    print(f"\nStatistics:")
    print(f"  Average original length: {original_lengths.mean():.0f} chars")
    print(f"  Average new length: {new_lengths.mean():.0f} chars")
    print(f"  Reduced by: {(1 - new_lengths.mean()/original_lengths.mean())*100:.1f}%")
    
    print(f"\nSample transformations:")
    for i in range(min(5, len(df))):
        orig_len = original_lengths.iloc[i]
        new_pred = df.iloc[i]['prediction']
        print(f"  Sample {i+1}: {orig_len} chars -> '{new_pred}'")
    
    print(f"\nSaving to {output_file}...")
    df.to_excel(output_file, index=False)
    print("Done!")
    
    # Also save as TSV for re-evaluation
    tsv_file = output_file.replace('.xlsx', '.tsv')
    df.to_csv(tsv_file, sep='\t', index=False)
    print(f"Also saved as {tsv_file}")

if __name__ == '__main__':
    main()
