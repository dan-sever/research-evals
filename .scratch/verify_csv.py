import csv
from collections import Counter

CSV_PATH = '/Users/dansever/Tavily Projects/Benchmarks/docs/tags/sealqa_seal0.csv'

import pyarrow.parquet as pq
t = pq.read_table('/Users/dansever/Tavily Projects/Benchmarks/data/sealqa_seal0.parquet')
parquet_qs = t.column('question').to_pylist()

rows = []
with open(CSV_PATH, newline='') as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    for r in reader:
        rows.append(r)

print('Columns:', cols)
print('Row count:', len(rows))

# Check q_index integrity
indices = [int(r['q_index']) for r in rows]
print('q_index range:', min(indices), '-', max(indices))
assert indices == list(range(111)), 'q_index ordering mismatch'

# Check questions match parquet exactly
mismatches = 0
for i, r in enumerate(rows):
    if r['question'] != parquet_qs[i]:
        print(f'MISMATCH at {i}:')
        print(f'  CSV:     {r["question"][:120]}')
        print(f'  PARQUET: {parquet_qs[i][:120]}')
        mismatches += 1
print(f'Question mismatches: {mismatches}')

# Validate label sets
reasoning_set = {'single-hop', 'multi-hop', 'comparative', 'unanswerable'}
retrieval_set = {'common', 'specialized', 'fresh', 'tricky-phrasing'}

bad = 0
for r in rows:
    if r['reasoning'] not in reasoning_set:
        print('Bad reasoning at', r['q_index'], ':', r['reasoning'])
        bad += 1
    if r['retrieval'] not in retrieval_set:
        print('Bad retrieval at', r['q_index'], ':', r['retrieval'])
        bad += 1
print(f'Invalid labels: {bad}')

# Distribution tables
n = len(rows)
print('\n=== Scheme A: reasoning ===')
c = Counter(r['reasoning'] for r in rows)
for label in ['single-hop', 'multi-hop', 'comparative', 'unanswerable']:
    cnt = c.get(label, 0)
    print(f'  {label:14s} {cnt:4d}  {100*cnt/n:5.1f}%')

print('\n=== Scheme B: retrieval ===')
c2 = Counter(r['retrieval'] for r in rows)
for label in ['common', 'specialized', 'fresh', 'tricky-phrasing']:
    cnt = c2.get(label, 0)
    print(f'  {label:16s} {cnt:4d}  {100*cnt/n:5.1f}%')

# Cross-tab
print('\n=== Cross-tab reasoning x retrieval ===')
xt = Counter((r['reasoning'], r['retrieval']) for r in rows)
print(f'{"":14s} {"common":>10} {"specialized":>12} {"fresh":>8} {"tricky":>8}')
for ra in ['single-hop', 'multi-hop', 'comparative', 'unanswerable']:
    row = [f'{ra:14s}']
    for rt in ['common', 'specialized', 'fresh', 'tricky-phrasing']:
        row.append(f'{xt.get((ra,rt),0):>10}' if rt != 'specialized' else f'{xt.get((ra,rt),0):>12}')
    print(' '.join(row))
