import json, collections

calls = [json.loads(l) for l in open("/tmp/opencode/grammar-fill/diag.jsonl")]


def stacks_of(c):
    s = c["s"]
    out, cur = [], []
    for w in s:
        if w == 0xFFFFFFFF:
            out.append(tuple(cur))
            cur = []
        else:
            cur.append(w)
    if cur:
        out.append(tuple(cur))
    return tuple(out)


state_keys = set()
stack_counts = collections.Counter()
for c in calls:
    st = stacks_of(c)
    state_keys.add(tuple(st))
    for one in st:
        stack_counts[one] += 1

print("calls:", len(calls))
print("distinct states (stack sets):", len(state_keys))
print("distinct stacks:", len(stack_counts))
print("total stack instances:", sum(stack_counts.values()))
print("stacks appearing once:", sum(1 for v in stack_counts.values() if v == 1))
print("top stacks:")
for st, n in stack_counts.most_common(5):
    pairs = [(st[i], st[i + 1]) for i in range(0, len(st), 2)]
    print("  n=%d pairs=%s" % (n, pairs))

# fills if masks were per-stack (union): distinct stacks needed
print()
print("if cached per stack: distinct stack masks to compute =", len(stack_counts))

# how many stacks per state?
cnt = collections.Counter(len(stacks_of(c)) for c in calls)
print("stacks-per-state histogram:", dict(cnt))
