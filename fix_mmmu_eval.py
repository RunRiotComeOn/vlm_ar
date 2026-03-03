#!/usr/bin/env python3
"""
Fix MMMU evaluation by extracting the final answer from verbose outputs
"""
import pandas as pd
import re
import sys
from pathlib import Path

def extract_final_answer(text, valid_choices):
    """
    Extract the final answer from verbose model output.
    Strategy: Look for the last occurrence of a valid choice letter
    """
    text = str(text)
    
    # Strategy 1: Look for explicit answer statement in last 500 chars
    last_part = text[-500:]
    patterns = [
        r'(?:answer|Answer|ANSWER)\s*(?:is|:)?\s*([A-Z])\b',
        r'(?:correct|Correct)\s+(?:answer|option)\s+(?:is|:)?\s*([A-Z])\b',
        r'(?:Therefore|So),?\s+(?:the\s+)?(?:answer|option)\s+(?:is|:)?\s*([A-Z])\b',
        r'\b([A-Z])\s*\.$',  # Single letter followed by period at end
        r'\n([A-Z])\s*$',     # Single letter at end after newline
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, last_part)
        if matches:
            for match in reversed(matches):  # Check from end
                if match in valid_choices:
                    return match
    
    # Strategy 2: Find all valid choice letters, return the last one
    all_matches = re.findall(r'\b([A-Z])\b', text)
    for match in reversed(all_matches):
        if match in valid_choices:
            # Make sure it's not just part of "option A" or similar
            # Look at context
            return match
    
    # Strategy 3: No clear answer found, return original for GPT matching
    return text

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_mmmu_eval.py <prediction_xlsx_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = input_file.replace('.xlsx', '_fixed.xlsx')
    
    print(f"Loading {input_file}...")
    df = pd.read_excel(input_file)
    
    print(f"Processing {len(df)} predictions...")
    
    # Determine valid choices for each question
    choice_cols = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    
    fixed_count = 0
    stats = {'original_len': [], 'fixed_len': [], 'fixed': []}
    
    for idx, row in df.iterrows():
        # Get valid choices for this question
        valid_choices = [col for col in choice_cols if col in row and pd.notna(row[col])]
        
        original = str(row['prediction'])
        fixed = extract_final_answer(original, valid_choices)
        
        df.at[idx, 'prediction'] = fixed
        
        stats['original_len'].append(len(original))
        stats['fixed_len'].append(len(fixed))
        stats['fixed'].append(len(fixed) < len(original))
        
        if len(fixed) < len(original):
            fixed_count += 1
    
    avg_orig = sum(stats['original_len']) / len(stats['original_len'])
    avg_fixed = sum(stats['fixed_len']) / len(stats['fixed_len'])
    
    print(f"\n统计:")
    print(f"  原始平均长度: {avg_orig:.0f} chars")
    print(f"  修复后平均长度: {avg_fixed:.0f} chars")
    print(f"  成功提取答案: {fixed_count} / {len(df)} ({fixed_count/len(df)*100:.1f}%)")
    print(f"  长度减少: {(1 - avg_fixed/avg_orig)*100:.1f}%")
    
    # Show some examples
    print(f"\n前5个修复示例:")
    for i in range(min(5, len(df))):
        orig_len = stats['original_len'][i]
        fixed_pred = df.iloc[i]['prediction']
        answer = df.iloc[i]['answer']
        is_correct = fixed_pred.strip() == answer.strip()
        print(f"  {i+1}. {orig_len} chars -> '{fixed_pred}' (正确答案: {answer}) {'✓' if is_correct else '✗'}")
    
    print(f"\n保存到 {output_file}...")
    df.to_excel(output_file, index=False)
    
    # Also save as TSV for VLMEvalKit
    tsv_file = output_file.replace('.xlsx', '.tsv')
    df.to_csv(tsv_file, sep='\t', index=False)
    print(f"也保存为 {tsv_file}")
    print("\n完成！现在可以用VLMEvalKit重新评估这个文件。")

if __name__ == '__main__':
    main()
