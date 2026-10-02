#!/usr/bin/env python3
"""Prototype probe for ticket #96: wrapper-grammar routes at the thinking boundary.

Runs against a LOCALLY served (patched, NINFER_PROTO_WRAPPER=1) NInfer on 127.0.0.1:8827.
Payload = graphiti CombinedExtraction messages/schema, same as probe3's G arm.

Arms:
  WA1..WA4  grammar = wrapper-A (full encoding, forced close)  enable_thinking=true
  WC1..WC4  grammar = wrapper-C (permissive prefix)            enable_thinking=true
  RB1..RB2  response_format json_schema, enable_thinking=false (criterion-4 regression)
  U1..U2    no constraint, enable_thinking=false               (criterion-4 regression)
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
OUT = os.environ.get('WRAPPER_OUT', '/tmp/opencode/proto-wrapper-e2e')
WRAP = os.environ.get('WRAPPER_GRAMMARS', '/tmp/opencode/proto-wrapper-cpu')
URL = os.environ.get('WRAPPER_URL', 'http://127.0.0.1:8827/v1/chat/completions')
RAW = os.path.join(OUT, 'raw')
os.makedirs(RAW, exist_ok=True)

pkg = types.ModuleType('gp')
pkg.__path__ = [ROOT]
sys.modules['gp'] = pkg


def load(name, fname):
    spec = importlib.util.spec_from_file_location('gp.' + name, os.path.join(ROOT, fname))
    m = importlib.util.module_from_spec(spec)
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
WRAPPERS = {name: open(os.path.join(WRAP, 'wrapper-%s.gbnf' % name)).read() for name in ('a', 'c')}

RUNS = []
for i in range(1, 5):
    RUNS.append(('WA%d' % i, 'a', 'wrapper'))
for i in range(1, 5):
    RUNS.append(('WC%d' % i, 'c', 'wrapper'))
for i in range(1, 3):
    RUNS.append(('RB%d' % i, None, 'schema'))
for i in range(1, 3):
    RUNS.append(('U%d' % i, None, 'plain'))


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


def salvage_reasoning_json(r):
    if not r:
        return None
    for start in (m.start() for m in reversed(list(re.finditer(r'\{', r)))):
        depth = 0
        for j in range(start, len(r)):
            if r[j] == '{':
                depth += 1
            elif r[j] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(r[start:j + 1])
                    except Exception:
                        break
    return None


results = []
for name, variant, mode in RUNS:
    payload = {'model': 'Qwen3.8-27B', 'messages': COMBINED, 'temperature': 1,
               'max_tokens': 4096}
    if mode == 'wrapper':
        payload['grammar'] = WRAPPERS[variant]
        payload['enable_thinking'] = True
    elif mode == 'schema':
        payload['response_format'] = {'type': 'json_schema',
                                      'json_schema': {'name': 'CombinedExtraction',
                                                      'schema': SCHEMA}}
        payload['enable_thinking'] = False
    status, body, dur = post(payload)
    with open(os.path.join(RAW, name + '.json'), 'w') as f:
        f.write(body)
    rec = dict(run=name, wrapper=variant, mode=mode, status=status, latency_s=round(dur, 2))
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
    rec['close_after'] = [reasoning[i + len('</think>'):i + len('</think>') + 1]
                          for i in re.finditer(re.escape('</think>'), reasoning)]
    rec['reasoning_head'] = reasoning[:160]
    if content:
        try:
            obj = json.loads(content)
            jsonschema.validate(obj, SCHEMA)
            rec['verdict'] = 'OK-content'
        except Exception as e:
            rec['verdict'] = 'content-bad: %s: %s' % (type(e).__name__, str(e)[:160])
    else:
        obj = salvage_reasoning_json(reasoning)
        if obj is not None:
            try:
                jsonschema.validate(obj, SCHEMA)
                rec['verdict'] = 'MISPLACED-in-reasoning (schema-valid)'
            except Exception as e:
                rec['verdict'] = 'MISPLACED-in-reasoning (schema-INVALID: %s)' % (str(e)[:120],)
        else:
            rec['verdict'] = 'EMPTY-no-json-found'
    results.append(rec)
    print('%s w=%s http=%s %5.1fs out=%-5s reas=%-5s content=%-5d close_after=%s %s' % (
        name, variant, status, dur, rec.get('out'), rec.get('reasoning_tokens'),
        rec.get('content_chars', -1), rec.get('close_after'), rec.get('verdict')), flush=True)

with open(os.path.join(OUT, 'results-wrapper.jsonl'), 'w') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')
print('\n=== done ===')
