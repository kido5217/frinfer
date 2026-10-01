import json, re, collections, os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- vocabulary ----
pieces = {}
for line in open(os.environ.get("VOCAB", "/tmp/opencode/grammar-fill/vocab.tsv")):
    id_, hexs = line.rstrip("\n").split("\t")
    pieces[int(id_)] = bytes.fromhex(hexs)


def cps(b):
    return b.decode("utf-8", "replace")


# JSON string char class from rule 0: NOT 0x22, CALT 0x5c, CALT 0x7f, RNG 0x00-0x1f
def class_match(cp):
    return cp != 0x22 and cp != 0x5C and cp != 0x7F and not (0x00 <= cp <= 0x1F)


def max_class_run(b):
    # maximal prefix consisting of class codepoints (escape-aware: \X counts as one match)
    s = b.decode("utf-8", "replace")
    run = 0
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            run += 1
            i += 2
            continue
        if class_match(ord(c)):
            run += 1
            i += 1
            continue
        break
    return run


runs = collections.Counter(max_class_run(b) for b in pieces.values())
print("class-run histogram (max 200):")
for k in sorted(runs)[:10]:
    print("  run", k, "->", runs[k])
print("  ...")
for k in sorted(runs)[-15:]:
    print("  run", k, "->", runs[k])
max_run = max(runs)
print("max class run:", max_run)


# ---- item chain body: "," space string traversals ----
def count_traversals(b):
    s = b.decode("utf-8", "replace")
    i = 0
    count = 0
    while i < len(s):
        if s[i] != ",":
            break
        j = i + 1
        # space: "" | " " | "\n"{1,2}[ \t]{0,20} -- generous skip
        while j < len(s) and s[j] in " \t\n":
            j += 1
        if j >= len(s) or s[j] != '"':
            break
        j += 1
        while j < len(s) and s[j] != '"':
            if s[j] == "\\" and j + 1 < len(s):
                j += 2
            else:
                j += 1
        if j >= len(s):
            break
        j += 1
        count += 1
        i = j
    return count


trav = collections.Counter(count_traversals(b) for b in pieces.values())
print("item-traversal histogram:", dict(trav))
max_trav = max(trav)
print("max item traversals:", max_trav)

# ---- load diag ----
calls = [
    json.loads(l)
    for l in open(
        os.environ.get("DIAG", os.path.join(HERE, "..", "results", "diary-diag.jsonl"))
    )
]
rules = {}
reps = {}
for line in open(
    os.environ.get("DIAG", os.path.join(HERE, "..", "results", "diary-diag.jsonl"))
    + ".rules"
):
    m = re.match(r"rule (\d+):(.*)", line)
    if m:
        rules[int(m.group(1))] = [
            (int(t), int(v)) for t, v in re.findall(r"\((\d+),(\d+)\)", m.group(2))
        ]
        continue
    m = re.match(r"rep (\d+) rule=(\d+) pos=(\d+)", line)
    if m:
        reps[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))

count = len(rules)
GLOBAL = 130


def body_sig(rule):
    elems = rules[rule]
    # frame: body [link]? ALT END ; body = elements before the optional trailing RULE_REF
    n = len(elems)
    assert elems[n - 1] == (0, 0) and elems[n - 2] == (1, 0), (rule, elems)
    end = n - 2
    if end >= 1 and elems[end - 1][0] == 2:  # trailing RULE_REF link
        end -= 1
    return tuple(elems[:end])


def threshold(body):
    if body == ((2, 0),):  # single REF to rule 0 (the string char class)
        return max_run
    if body == ((2, 6),):  # "," space string
        return max(1, max_trav)
    return GLOBAL


def canon(atom):
    rid, idx = atom
    if rid in reps:
        rule, pos = reps[rid]
        th = threshold(body_sig(rule))
        return (("F", body_sig(rule), min(pos, th)), idx)
    return (("R", rid), idx)


def key_atoms(c):
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
    atoms = []
    for stack in out:
        for i in range(0, len(stack), 2):
            atoms.append((stack[i], stack[i + 1]))
    return tuple(sorted(atoms))


# replay: folds + fills
seen = {}
fills = 0
hits = 0
seen_masks = {}
for c in calls:
    k = tuple(sorted(canon(a) for a in key_atoms(c)))
    if k in seen:
        hits += 1
    else:
        seen[k] = True
        fills += 1
        seen_masks.setdefault(c["m"], len(seen_masks))

print("\nbaseline calls:", len(calls))
print(
    "with per-body thresholds: distinct folded keys =",
    len(seen),
    " fills =",
    fills,
    " hits =",
    hits,
)
print("distinct masks (unchanged):", len(seen_masks))

# also count distinct keys with the current global fold (sanity: should match diag fills=1957)
seen2 = {}
fills2 = 0
for c in calls:
    k = tuple(
        sorted(
            (
                (
                    ("R", a[0])
                    if a[0] not in reps
                    else ("F", body_sig(reps[a[0]][0]), min(reps[a[0]][1], GLOBAL))
                ),
                a[1],
            )
            for a in key_atoms(c)
        )
    )
    if k in seen2:
        continue
    seen2[k] = True
    fills2 += 1
print("current global fold: distinct keys =", len(seen2), "fills =", fills2)
