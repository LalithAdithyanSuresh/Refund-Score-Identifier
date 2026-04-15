import pandas as pd

df = pd.read_csv('validation_results.csv')

total = len(df)
valid = len(df[df['decision'] == 'VALID'])
reject = len(df[df['decision'] == 'REJECT'])

print(f"Total: {total}, VALID: {valid} ({valid/total*100:.1f}%), REJECT: {reject} ({reject/total*100:.1f}%)")
print()

rejected = df[df['decision'] == 'REJECT']

def cat(r):
    if 'Brand mismatch' in r: return 'Brand Mismatch'
    if 'Color mismatch' in r: return 'Color Mismatch'
    if 'Outside refund' in r: return 'Outside Window'
    if 'Change of mind' in r: return 'Change of Mind'
    if 'Vague' in r: return 'Vague'
    return r

cats = rejected['reason'].apply(cat).value_counts()
print("REJECTION BREAKDOWN:")
for c, n in cats.items():
    print(f"  {c:25s}: {n}")

# Check for false rejections - show ALL rejected
print("\n" + "="*80)
print("ALL REJECTED COMPLAINTS:")
print("="*80)
for _, row in rejected.iterrows():
    print(f"\n  ID={row['complaint_id']:3d} [{cat(row['reason'])}]")
    print(f"  Text: {row['complaint_text'][:100]}")
    print(f"  Reason: {row['reason'][:120]}")

# Check for false accepts - look at change-of-mind texts that are VALID
print("\n" + "="*80)
print("CHECKING CHANGE-OF-MIND THAT MIGHT BE FALSE VALID:")
print("="*80)
valid_df = df[df['decision'] == 'VALID']
com_keywords = ['changed my mind', 'prefer another', 'dont like', 'dont need', 'want to return', 'want to exchange']
for _, row in valid_df.iterrows():
    text_low = row['complaint_text'].lower()
    for kw in com_keywords:
        if kw in text_low:
            print(f"\n  ID={row['complaint_id']:3d}: {row['complaint_text'][:100]}")
            print(f"  Decision: {row['decision']}")
            break

# Check vague that are VALID
print("\n" + "="*80)
print("CHECKING VAGUE TEXTS THAT MIGHT BE FALSE VALID:")
print("="*80)
vague_keywords = ['bad product', 'not good', 'terrible', 'worst', 'unhappy', 'not satisfied', 'want my money back']
for _, row in valid_df.iterrows():
    text_low = row['complaint_text'].lower()
    for kw in vague_keywords:
        if kw in text_low:
            print(f"\n  ID={row['complaint_id']:3d}: {row['complaint_text'][:100]}")
            print(f"  Decision: {row['decision']}")
            break
