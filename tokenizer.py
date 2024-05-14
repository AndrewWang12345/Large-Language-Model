with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
tokens = text.encode("utf-8")
tokens = list(map(int, tokens))
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair,0)+1
    return counts
stats = get_stats(tokens)
top_pair = max(stats,key=stats.get)
def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1]==pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[1])
            i += 1
    return newids
desire_vocab_size = 276
num_merges = desire_vocab_size - 256
ids = list(tokens)
merges = {}
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    ids = merge(ids, pair, idx)
    merges[pair] = idx