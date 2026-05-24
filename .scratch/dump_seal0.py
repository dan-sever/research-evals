import pyarrow.parquet as pq

t = pq.read_table('/Users/dansever/Tavily Projects/Benchmarks/data/sealqa_seal0.parquet')
qs = t.column('question').to_pylist()
ans = t.column('answer').to_pylist()

lines = []
for i, (q, a) in enumerate(zip(qs, ans)):
    q1 = q.replace('\n', ' ').replace('\r', ' ')
    a1 = (a or '').replace('\n', ' ').replace('\r', ' ')[:400]
    lines.append(f'{i}|||{q1}|||A:{a1}')

with open('/Users/dansever/Tavily Projects/Benchmarks/.scratch/seal0_qs.txt', 'w') as f:
    f.write('\n'.join(lines) + '\n')

print('wrote', len(lines), 'rows')
