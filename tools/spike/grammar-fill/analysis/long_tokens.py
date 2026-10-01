import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
pieces = {}
for line in open(os.environ.get("VOCAB", "/tmp/opencode/grammar-fill/vocab.tsv")):
    id_, hexs = line.rstrip("\n").split("\t")
    pieces[int(id_)] = bytes.fromhex(hexs)


def class_match(cp):
    return cp != 0x22 and cp != 0x5C and cp != 0x7F and not (0x00 <= cp <= 0x1F)


def max_class_run(b):
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


top = sorted(pieces.items(), key=lambda kv: -max_class_run(kv[1]))[:25]
for id_, b in top:
    s = b.decode("utf-8", "replace")
    print(id_, max_class_run(b), repr(s[:60]), "bytes:", b[:40].hex())

# Which ids are EOG per generation_config?
import sys

if len(sys.argv) < 2:
    print("usage: long_tokens.py <tokenizer-snapshot-dir>", file=sys.stderr)
    sys.exit(2)
snap = sys.argv[1]
gc = json.load(open(snap + "/generation_config.json"))
print("generation_config eos:", gc.get("eos_token_id"))

# added tokens: check tokenizer.json added_tokens
tok = json.load(open(snap + "/tokenizer.json"))
added = tok.get("added_tokens", [])
print("added tokens count:", len(added))
for a in added:
    print("  added id", a["id"], repr(a["content"]), "special", a.get("special"))
