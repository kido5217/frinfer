import json, re, collections

import os
HERE = os.path.dirname(os.path.abspath(__file__))
calls = [json.loads(l) for l in open(os.environ.get("DIAG", os.path.join(HERE, "..", "results", "diary-diag.jsonl")))]
rules = {}
reps = {}
for line in open(os.environ.get("DIAG", os.path.join(HERE, "..", "results", "diary-diag.jsonl")) + ".rules"):
    m = re.match(r"rule (\d+):(.*)", line)
    if m:
        rules[int(m.group(1))] = [
            (int(t), int(v)) for t, v in re.findall(r"\((\d+),(\d+)\)", m.group(2))
        ]
        continue
    m = re.match(r"rep (\d+) rule=(\d+) pos=(\d+)", line)
    if m:
        reps[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))


def body_sig(rule):
    elems = rules[rule]
    n = len(elems)
    end = n - 2
    if end >= 1 and elems[end - 1][0] == 2:
        end -= 1
    return tuple(elems[:end])


def atoms_of(c):
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
    a = []
    for stack in out:
        for i in range(0, len(stack), 2):
            a.append((stack[i], stack[i + 1]))
    return tuple(sorted(a))


def atom_info(a):
    rid, idx = a
    if rid in reps:
        rule, pos = reps[rid]
        return ("F", body_sig(rule), idx, pos)
    return ("R", rid, idx, None)


contexts = collections.defaultdict(list)
for idx, c in enumerate(calls):
    atoms = [atom_info(a) for a in atoms_of(c)]
    for slot, a in enumerate(atoms):
        if a[0] != "F":
            continue
        ctx = tuple(atoms[:slot] + atoms[slot + 1 :])
        contexts[(ctx, a[1], a[2])].append((a[3], c["m"], idx))

for (ctx, body, bidx), entries in contexts.items():
    if body != ((2, 6),):
        continue
    positions = {e[0] for e in entries}
    masks = {e[1] for e in entries}
    if len(positions) > 1 and len(masks) > 1:
        print("BODY", body, "element idx", bidx, "positions", sorted(positions))
        for p, m, i in sorted(entries):
            print("   pos", p, "mask", m, "call", i)
        print("context (non-item atoms):")
        for a in ctx:
            print("   ", a)
        print()
