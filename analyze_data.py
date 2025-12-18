import pandas as pd
import numpy as np

train_df = pd.read_csv('data/prepared/train.csv')
val_df = pd.read_csv('data/prepared/val.csv')

print('=' * 60)
print('TRAIN SET ANALYSIS')
print('=' * 60)
print(f'Total samples: {len(train_df)}')
print('\nScore distributions:')
for col in ['overall', 'intonation', 'tone', 'timing', 'technique']:
    print(f'\n{col}:')
    print(f'  Mean: {train_df[col].mean():.2f}')
    print(f'  Std:  {train_df[col].std():.2f}')
    print(f'  Min:  {train_df[col].min()}')
    print(f'  Max:  {train_df[col].max()}')
    print(f'  Value counts:')
    counts = train_df[col].value_counts().sort_index()
    for val_score, count in counts.items():
        print(f'    {val_score}: {count} ({count/len(train_df)*100:.1f}%)')

print('\n' + '=' * 60)
print('VAL SET ANALYSIS')
print('=' * 60)
print(f'Total samples: {len(val_df)}')
print('\nScore distributions:')
for col in ['overall', 'intonation', 'tone', 'timing', 'technique']:
    print(f'\n{col}:')
    print(f'  Mean: {val_df[col].mean():.2f}')
    print(f'  Std:  {val_df[col].std():.2f}')
    print(f'  Min:  {val_df[col].min()}')
    print(f'  Max:  {val_df[col].max()}')

print('\n' + '=' * 60)
print('POTENTIAL ISSUES')
print('=' * 60)

# Check for issues
if len(train_df) < 200:
    print(f'⚠️  Small dataset: Only {len(train_df)} training samples')
    print('   This is very small for deep learning. Consider:')
    print('   - Data augmentation')
    print('   - Transfer learning')
    print('   - Simpler model')

if len(val_df) < 20:
    print(f'⚠️  Very small validation set: Only {len(val_df)} samples')
    print('   Validation metrics may be unreliable')

# Check score variance
low_variance_scores = []
for col in ['overall', 'intonation', 'tone', 'timing', 'technique']:
    if train_df[col].std() < 1.0:
        low_variance_scores.append(col)
        print(f'⚠️  Low variance in {col}: std={train_df[col].std():.2f}')
        print('   Model may struggle to learn meaningful patterns')

# Check for class imbalance
for col in ['overall', 'intonation', 'tone', 'timing', 'technique']:
    counts = train_df[col].value_counts()
    max_count = counts.max()
    min_count = counts.min()
    if max_count / min_count > 5:
        print(f'⚠️  Severe class imbalance in {col}:')
        print(f'   Most common: {counts.idxmax()} ({max_count} samples)')
        print(f'   Least common: {counts.idxmin()} ({min_count} samples)')
        print(f'   Ratio: {max_count/min_count:.1f}x')

