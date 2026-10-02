#!/usr/bin/env python3
"""Composition probe for ticket #97: constrained close rule x #96 wrapper A.

Runs against the locally served build carrying BOTH prototype patches
(NINFER_PROTO_WRAPPER=1 engages wrapper A's mask from token 0; the request also
carries the constrained-mode close rule signal). 4 runs of wrapper A with
enable_thinking=true; expectation from #96: 4/4 schema-valid non-empty content,
free-form reasoning, marker consumed, whitespace decider.
"""
import importlib.util
import json
import os
import re
import sys
import time
import types
import urllib.error
import urllib.request

import jsonschema

ROOT = '/tmp/opencode/graphiti-probe/graphiti/graphiti_core/prompts'
E2E = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('COMPOSE_OUT', '/tmp/opencode/proto-close-rule-e2e/compose')
URL = os.environ.get('CLOSE_RULE_URL', 'http://127.0.0.1:8827/v1/chat/completions')
RAW = os.path.join(OUT, 'raw')
os.makedirs(RAW, exist_ok=True)

pkg = types.ModuleType('gp')
pkg.__path__ = [ROOT]
sys.modules['gp'] = pkg


def load(name, fname):
    spec = importlib.util.spec_from_file_location('gp.' + name, os.path.join(ROOT, fname))
    m = types.ModuleType('gp.' + name)
    sys.modules['gp.' + name] = m
    spec.loader.exec_module(m)
    return m


gc = types.ModuleType('graphiti_core')
gc.__path__ = []
gcu = types.ModuleType('graphiti_core.utils')
gcu.__path__ = []
gcut = types.ModuleType('graphiti_core.utils.text_utils')
gcut.MAX_SUMMARY_CHARS = 100000
sys.modules.update({'graphiti_core': gc, 'graphiti_core.utils': gcu,
                    'graphiti_core.utils.text_utils': gcut})

load('models', 'models.py')
load('prompt_helpers', 'prompt_helpers.py')
ene = load('extract_nodes_and_edges', 'extract_nodes_and_edges.py')
SCHEMA = ene.CombinedExtraction.model_json_schema()

ENTITY_TYPES_CONTEXT = [{
    'entity_type_id': 0, 'entity_type_name': 'Entity',
    'entity_type_description': (
        'A specific, identifiable entity that does not fit any of the other listed '
        'types. Must still be a concrete, meaningful thing — specific enough to be '
        'uniquely identifiable. GOOD: a named entity not covered by the other types. '
        'BAD: "luck", "ideas", "tomorrow", "things", "them", "everybody", '
        '"a sense of wonder", "great times". When in doubt, do not extract the entity.'
    )}]
EPISODE = """[Episode 0]
User: Hey, I just got back from my cycling trip around Lake Baikal — the weather was perfect. My new road bike, the Canyon Ultimate CF, made the long climbs way easier.
Assistant: That sounds amazing! What was the hardest stage?
User: The north loop, about 120 km. I stayed at the Angara River B&B near Irkutsk and ate incredible borscht at the restaurant there. Next I'm planning a trip to the Altai mountains in July."""
LANG_INSTR = (
    '\n\nAny extracted information should be returned in the same language as it was '
    'written in. Only output non-English text when the user has written full sentences '
    'or phrases in that non-English language. Otherwise, output English.')


def build_combined(episode):
    context = {'episode_content': episode, 'previous_episodes': [],
               'custom_extraction_instructions': '', 'entity_types': ENTITY_TYPES_CONTEXT,
               'edge_types': []}
    msgs = ene.extract_message(context)
    out = []
    for i, m in enumerate(msgs):
        c = m.content + (LANG_INSTR if i == 0 else '')
        out.append({'role': m.role, 'content': c})
    return out


COMBINED = build_combined(EPISODE)
WRAPPER_A = open(os.path.join(E2E, 'wrapper-a.gbnf')).read()


def post(payload, timeout=600):
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), time.time() - t0
    except Exception as e:
        return -1, repr(e), time.time() - t0


results = []
for i in range(1, 5):
    name = 'WA%d_compose' % i
    payload = {'model': 'Qwen3.8-27B', 'messages': COMBINED, 'temperature': 1,
               'max_tokens': 4096, 'grammar': WRAPPER_A, 'enable_thinking': True}
    status, body, dur = post(payload)
    with open(os.path.join(RAW, name + '.json'), 'w') as f:
        f.write(body)
    rec = dict(run=name, status=status, latency_s=round(dur, 2))
    if status != 200:
        rec['note'] = 'HTTP %s: %s' % (status, body[:300])
        results.append(rec)
        print(rec, flush=True)
        continue
    j = json.loads(body)
    m = j['choices'][0]['message']
    u = j.get('usage') or {}
    rd = u.get('completion_tokens_details') or {}
    content = m.get('content') or ''
    reasoning = m.get('reasoning_content') or ''
    rec['finish'] = j['choices'][0].get('finish_reason')
    rec['out'] = u.get('completion_tokens')
    rec['reasoning_tokens'] = rd.get('reasoning_tokens', 0)
    rec['content_chars'] = len(content)
    rec['reasoning_chars'] = len(reasoning)
    rec['close_marker_in_reasoning'] = '</think>' in reasoning
    rec['decider'] = [reasoning[i + len('</think>'):i + len('</think>') + 1]
                      for i in re.finditer(re.escape('</think>'), reasoning)]
    if content:
        try:
            obj = json.loads(content)
            rec['keys'] = list(obj.keys())[:6]
            jsonschema.validate(obj, SCHEMA)
            rec['verdict'] = 'OK-content schema-valid'
        except Exception as e:
            rec['verdict'] = 'content-bad: %s: %s' % (type(e).__name__, str(e)[:200])
    else:
        rec['verdict'] = 'EMPTY'
    results.append(rec)
    print('%s http=%s %5.1fs out=%-5s reas=%-5s content=%-5d keys=%-25s %s' % (
        name, status, dur, rec.get('out'), rec.get('reasoning_tokens'),
        rec.get('content_chars', -1), str(rec.get('keys', ''))[:25],
        rec.get('verdict', rec.get('note', ''))), flush=True)

with open(os.path.join(OUT, 'results.jsonl'), 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')
print('\n=== done ===')
