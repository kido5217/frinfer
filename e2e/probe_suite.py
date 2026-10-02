#!/usr/bin/env python3
"""Live e2e probe suite: opencode request shapes against the deployed ninfer-yarn serve.

Wayfinder map #139, ticket #143. Each probe replays a request shape that opencode
v2.0.22 actually sends (see docs/research/opencode-requirements.md on branch
research/opencode-requirements) or a documented serve failure shape (see
docs/research/serve-capabilities.md on branch research/serve-capabilities),
against the deployed unit `ninfer-yarn-qwen38-27b` at 127.0.0.1:8827.

Stdlib only (urllib); run as:  python3 e2e/probe_suite.py [--out DIR] [--base URL]

Probes are run sequentially and generate SHORT outputs (small max_completion_tokens)
except where noted; the total GPU footprint is modest. The overflow probe (P16)
is rejected at submit (no generation) but tokenizes a ~3.5 MB prompt on the CPU.
"""

import json
import re
import sys
import time
import urllib.request
import urllib.error
import base64
import gzip
import io

BASE = "http://127.0.0.1:8827/v1"
MODEL = "Qwen3.8-27B"

# opencode v2.0.22 sends these headers on every request (companion doc §1.4).
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "opencode/cli/2.0.22/opencode",
    "x-opencode-session-id": "ses_e2e143",
    "x-opencode-project": "proj_e2e143",
    "x-opencode-client": "opencode",
    "x-session-affinity": "e2e143",
}

# 1x1 red PNG (data URL) — pipeline check for vision, not a semantics check.
TINY_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
)


def post_chat(body, base=None, timeout=600):
    """POST one chat completion; return (status, payload-or-sse-text, headers)."""
    url = (base or BASE) + "/chat/completions"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=dict(HEADERS))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace"), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), dict(e.headers)


def get(path, base=None, timeout=30):
    url = (base or BASE).rsplit("/v1", 1)[0] + path
    req = urllib.request.Request(url, headers=dict(HEADERS))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def parse_sse(text):
    """Parse SSE text into a list of (event_json_or_marker) in order."""
    events = []
    for line in text.splitlines():
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                events.append("[DONE]")
            elif payload == "null" or payload == "":
                continue  # tolerated flush markers (companion doc §3.1)
            else:
                events.append(json.loads(payload))
        elif line.startswith(":"):
            continue  # comment / keep-alive
    return events


def sse_contract(events):
    """Check the opencode stream contract (companion doc §3) on parsed events.
    Returns list of (check, expected, observed, pass)."""
    checks = []
    done = events[-1] == "[DONE]" if events else False
    checks.append(("stream ends with [DONE]", True, done, done))
    if not done:
        return checks
    js = events[:-1]
    if not js:
        checks.append(("at least one JSON event", True, False, False))
        return checks
    first = js[0]
    checks.append(
        (
            "start chunk delta.role == assistant",
            "assistant",
            first.get("choices", [{}])[0].get("delta", {}).get("role"),
            first.get("choices", [{}])[0].get("delta", {}).get("role") == "assistant",
        )
    )
    finish_idx = None
    for i, ev in enumerate(js):
        for ch in ev.get("choices", []):
            if ch.get("finish_reason"):
                finish_idx = i
    fr_ok = finish_idx is not None
    fr = js[finish_idx]["choices"][0]["finish_reason"] if fr_ok else None
    checks.append(
        ("finish_reason present before [DONE]", "yes", fr or "absent", fr_ok)
    )
    if fr_ok:
        checks.append(
            ("finish_reason in accepted set {stop,length,tool_calls}",
             "stop|length|tool_calls", fr, fr in ("stop", "length", "tool_calls"))
        )
    bad_after = False
    if finish_idx is not None:
        for ev in js[finish_idx + 1:]:
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if any(d.get(k) not in (None, "") for k in ("content", "reasoning_content", "reasoning", "tool_calls")):
                    bad_after = True
    checks.append(("no content after finish chunk", "none", "present" if bad_after else "none", not bad_after))
    usage_ev = None
    for ev in reversed(js):
        if ev.get("usage"):
            usage_ev = ev
            break
    u = (usage_ev or {}).get("usage", {})
    need = ["prompt_tokens", "completion_tokens", "total_tokens"]
    missing = [k for k in need if k not in u]
    checks.append(("usage chunk present with core fields", "prompt/completion/total",
                   "missing: " + ",".join(missing) if missing else "ok", not missing))
    det_p = u.get("prompt_tokens_details", {})
    det_c = u.get("completion_tokens_details", {})
    checks.append(
        ("usage carries prompt_tokens_details.cached_tokens and completion_tokens_details.reasoning_tokens",
         "both", ("cached" if "cached_tokens" in det_p else "") + "/(" +
                 ("reasoning" if "reasoning_tokens" in det_c else "") + ")",
         "cached_tokens" in det_p and "reasoning_tokens" in det_c)
    )
    single_choice = all(len(ev.get("choices", [])) <= 1 for ev in js)
    checks.append(("exactly one choice per event", True, single_choice, single_choice))
    return checks


def tool_call_checks(events):
    """Check the tool-call wire contract (companion doc §3.3) on a parsed stream."""
    checks = []
    tc_events = []
    for ev in events:
        if isinstance(ev, str):
            continue
        for ch in ev.get("choices", []):
            d = ch.get("delta", {})
            if d.get("tool_calls"):
                tc_events.append(d["tool_calls"])
    checks.append(("tool_calls appear", ">=1 delta", len(tc_events), len(tc_events) > 0))
    if tc_events:
        complete = False
        if len(tc_events) == 1 and all(len(t) >= 1 for t in tc_events):
            for t in tc_events[0]:
                ok = (
                    re.fullmatch(r"call_[0-9a-f]{16}", t.get("id") or "") is not None
                    and t.get("type") == "function"
                    and bool((t.get("function") or {}).get("name"))
                )
                try:
                    args = (t.get("function") or {}).get("arguments") or "{}"
                    json.loads(args)
                    ok = ok and True
                except Exception:
                    ok = False
                complete = complete and ok
        checks.append(
            ("single terminal tool_calls delta with complete id+name+valid-JSON arguments",
             "one complete delta",
             f"{len(tc_events)} delta(s), complete={complete}", complete)
        )
    return checks


def common_body(messages, **extra):
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "store": False,
        "max_completion_tokens": extra.pop("mct", 512),
        "max_tokens": extra.pop("mt", 32768),
    }
    body.update(extra)
    return body


SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Execute a shell command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to run."}
            },
            "required": ["command"],
        },
        "strict": False,
    },
}


def run(id, name, fn):
    t0 = time.time()
    try:
        out = fn()
    except Exception as e:  # a probe raising is itself a finding
        out = {"checks": [("probe exception", "no exception", repr(e), False)],
               "verdict": "fail", "raw_head": "", "raw_tail": ""}
    elapsed_ms = int((time.time() - t0) * 1000)
    verdict = "pass" if all(c[3] for c in out["checks"]) else (
        "fail" if any(c[0] for c in out["checks"]) else "error")
    rec = {
        "id": id, "name": name, "verdict": verdict, "wall_ms": elapsed_ms,
        "checks": out["checks"],
        "raw_head": out.get("raw_head", "")[:1500],
        "raw_tail": out.get("raw_tail", "")[-1500:],
    }
    return rec


def main():
    out_dir = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "e2e/results"
    base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else BASE
    import os
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results_path = f"{out_dir}/probes-{stamp}.jsonl"
    recs = []

    def add(id, name, fn):
        rec = run(id, name, fn)
        recs.append(rec)
        print(f"{id} {name}: {rec['verdict']} ({rec['wall_ms']} ms)", flush=True)
        return rec

    def probe_models():
        st, txt = get("/v1/models")
        ok_status = st == 200
        j = json.loads(txt) if ok_status else {}
        d0 = (j.get("data") or [{}])[0]
        return {"checks": [
            ("GET /v1/models -> 200", 200, st, ok_status),
            ("model id == Qwen3.8-27B", MODEL, d0.get("id"), d0.get("id") == MODEL),
            ("max_model_len == 446902 (YaRN window)", 446902, d0.get("max_model_len"),
             d0.get("max_model_len") == 446902),
        ]}

    def probe_health():
        st, txt = get("/health")
        return {"checks": [("GET /health -> 200 {status:ok}", 200, st, st == 200 and json.loads(txt).get("status") == "ok")]}

    def probe_text_stream():
        body = common_body(
            [{"role": "system", "content": "You are a concise assistant."},
             {"role": "user", "content": "Reply with exactly: OK"}],
            mct=128, reasoning_effort="xhigh")
        st, txt, hdrs = post_chat(body)
        events = parse_sse(txt) if st == 200 else []
        checks = [("POST -> 200", 200, st, st == 200)]
        if st == 200:
            checks += sse_contract(events)
            has_reasoning = any(
                isinstance(e, dict) and any(
                    (ch.get("delta") or {}).get("reasoning_content")
                    for ch in e.get("choices", [])) for e in events)
            checks.append(("reasoning_content deltas present (thinking on)", True, has_reasoning, has_reasoning))
        return {"checks": checks, "raw_head": txt[:1500], "raw_tail": txt[-1500:]}

    def probe_max_precedence():
        # mirrors the repo test pin: max_completion_tokens 48 + max_tokens 8 -> 48 wins
        body = common_body(
            [{"role": "user", "content": "Count up from 1 to 100, one number per line. No commentary."}],
            mct=48, mt=8)
        st, txt, _ = post_chat(body)
        events = parse_sse(txt) if st == 200 else []
        checks = [("POST -> 200", 200, st, st == 200)]
        if st == 200:
            checks += sse_contract(events)
            u = {}
            for e in reversed(events):
                if isinstance(e, dict) and e.get("usage"):
                    u = e["usage"]
                    break
            n = u.get("completion_tokens")
            checks.append(("completion bounded by max_completion_tokens=48 (not max_tokens=8)",
                           "8 < n <= 48", n, isinstance(n, int) and 8 < n <= 48))
            fr = None
            for e in events:
                if isinstance(e, dict):
                    for ch in e.get("choices", []):
                        if ch.get("finish_reason"):
                            fr = ch["finish_reason"]
            checks.append(("finish_reason == length", "length", fr, fr == "length"))
        return {"checks": checks, "raw_head": txt[:1500], "raw_tail": txt[-1500:]}

    def probe_no_max_default():
        body = common_body(
            [{"role": "user", "content": "Reply with the single word: ping"}], mct=None, mt=None)
        body.pop("max_completion_tokens", None)
        body.pop("max_tokens", None)
        st, txt, _ = post_chat(body)
        events = parse_sse(txt) if st == 200 else []
        checks = [("POST without any max field -> 200", 200, st, st == 200)]
        if st == 200:
            checks += sse_contract(events)
        return {"checks": checks, "raw_head": txt[:1500], "raw_tail": txt[-1500:]}

    def probe_tool_call():
        body = common_body(
            [{"role": "system", "content": "You are a CLI agent. Use the provided tools."},
             {"role": "user", "content": "Use the shell tool to run exactly: echo e2e143-tool && pwd . "
                                          "Respond with only the tool call."}],
            mct=1024, tools=[SHELL_TOOL], reasoning_effort="xhigh")
        st, txt, _ = post_chat(body)
        events = parse_sse(txt) if st == 200 else []
        checks = [("POST -> 200", 200, st, st == 200)]
        if st == 200:
            checks += sse_contract(events)
            checks += tool_call_checks(events)
            fr = None
            for e in events:
                if isinstance(e, dict):
                    for ch in e.get("choices", []):
                        if ch.get("finish_reason"):
                            fr = ch["finish_reason"]
            checks.append(("finish_reason == tool_calls", "tool_calls", fr, fr == "tool_calls"))
        return {"checks": checks, "raw_head": txt[:2000], "raw_tail": txt[-2000:]}

    def probe_tool_replay():
        # opencode replays assistant tool turns with content null + reasoning_content
        msgs = [
            {"role": "system", "content": "You are a CLI agent. Use the provided tools."},
            {"role": "user", "content": "Run: echo first && echo second"},
            {"role": "assistant", "content": None, "reasoning_content": "run the two echoes",
             "tool_calls": [{"id": "call_" + "ab" * 8, "type": "function",
                             "function": {"name": "shell",
                                          "arguments": "{\"command\":\"echo first && echo second\"}"}}]},
            {"role": "tool", "tool_call_id": "call_" + "ab" * 8, "content": "first\nsecond"},
            {"role": "user", "content": "Now run: echo third. Respond with only the tool call."},
        ]
        body = common_body(msgs, mct=1024, tools=[SHELL_TOOL], reasoning_effort="xhigh")
        st, txt, _ = post_chat(body)
        events = parse_sse(txt) if st == 200 else []
        checks = [("POST with tool history replay -> 200 (no 400 on content:null / reasoning_content)",
                   200, st, st == 200)]
        if st == 200:
            checks += sse_contract(events)
            checks += tool_call_checks(events)
        return {"checks": checks, "raw_head": txt[:2000], "raw_tail": txt[-2000:]}

    def probe_tool_choice_none():
        body = common_body(
            [{"role": "user", "content": "Reply with the single word: none"}],
            mct=128, tools=[SHELL_TOOL], tool_choice="none")
        st, txt, _ = post_chat(body)
        events = parse_sse(txt) if st == 200 else []
        checks = [("tool_choice none -> 200", 200, st, st == 200)]
        if st == 200:
            checks += sse_contract(events)
            checks += [("no tool_calls emitted", "none",
                        any(isinstance(e, dict) and any(
                            (ch.get("delta") or {}).get("tool_calls") for ch in e.get("choices", []))
                            for e in events),
                        not any(isinstance(e, dict) and any(
                            (ch.get("delta") or {}).get("tool_calls") for ch in e.get("choices", []))
                            for e in events))]
        return {"checks": checks, "raw_head": txt[:1500], "raw_tail": txt[-1500:]}

    def expect_reject(id, name, body, code, status_expected=400):
        st, txt, _ = post_chat(body)
        j = {}
        try:
            j = json.loads(txt)
        except Exception:
            pass
        err = j.get("error", {})
        checks = [
            (f"HTTP {status_expected}", status_expected, st, st == status_expected),
            ("error.code", code, err.get("code"), err.get("code") == code),
            ("error.type invalid_request_error (400) or envelope present",
             "envelope", "error" in j, "error" in j),
        ]
        return {"checks": checks, "raw_head": txt[:800], "raw_tail": txt[-400:]}

    def probe_tool_choice_required():
        body = common_body([{"role": "user", "content": "hello"}], mct=128,
                           tools=[SHELL_TOOL], tool_choice="required")
        return expect_reject("P09", "tool_choice required", body, "tool_choice_not_supported")

    def probe_parallel_false():
        body = common_body([{"role": "user", "content": "hello"}], mct=128,
                           tools=[SHELL_TOOL], parallel_tool_calls=False)
        return expect_reject("P10", "parallel_tool_calls false", body, "parallel_tool_calls_not_supported")

    def probe_strict_true():
        tool = json.loads(json.dumps(SHELL_TOOL))
        tool["function"]["strict"] = True
        body = common_body([{"role": "user", "content": "hello"}], mct=128, tools=[tool])
        return expect_reject("P11", "strict true", body, "strict_tools_not_supported")

    def probe_store_true():
        body = common_body([{"role": "user", "content": "hello"}], mct=128)
        body["store"] = True
        return expect_reject("P12", "store true", body, "store_not_supported")

    def probe_model_mismatch():
        body = common_body([{"role": "user", "content": "hello"}], mct=128)
        body["model"] = "wrong-model"
        st, txt, _ = post_chat(body)
        j = {}
        try:
            j = json.loads(txt)
        except Exception:
            pass
        err = j.get("error", {})
        return {"checks": [
            ("HTTP 404 (not 400)", 404, st, st == 404),
            ("error.code model_not_found", "model_not_found", err.get("code"),
             err.get("code") == "model_not_found"),
        ], "raw_head": txt[:800], "raw_tail": txt[-400:]}

    def probe_unknown_field():
        body = common_body([{"role": "user", "content": "Reply with the single word: ok"}], mct=64)
        body["future_unknown_field"] = "x"
        st, txt, _ = post_chat(body)
        events = parse_sse(txt) if st == 200 else []
        checks = [("unknown top-level field tolerated -> 200", 200, st, st == 200)]
        if st == 200:
            checks += sse_contract(events)
        return {"checks": checks, "raw_head": txt[:1500], "raw_tail": txt[-1500:]}

    def probe_effort_none():
        body = common_body(
            [{"role": "user", "content": "Reply with the single word: ok"}],
            mct=64, reasoning_effort="none")
        st, txt, _ = post_chat(body)
        events = parse_sse(txt) if st == 200 else []
        checks = [("reasoning_effort none -> 200", 200, st, st == 200)]
        if st == 200:
            checks += sse_contract(events)
            has_reasoning = any(
                isinstance(e, dict) and any(
                    (ch.get("delta") or {}).get("reasoning_content")
                    for ch in e.get("choices", [])) for e in events)
            checks.append(("no reasoning_content deltas (thinking disabled)", False, has_reasoning, not has_reasoning))
        return {"checks": checks, "raw_head": txt[:1500], "raw_tail": txt[-1500:]}

    def probe_overflow():
        # > 446902 prompt tokens -> 400 context_length_exceeded at submit
        body = common_body(
            [{"role": "user", "content": "e2e143 " * 500000}], mct=64)
        st, txt, _ = post_chat(body, timeout=300)
        j = {}
        try:
            j = json.loads(txt)
        except Exception:
            pass
        err = j.get("error", {})
        msg = err.get("message", "")
        checks = [
            ("HTTP 400", 400, st, st == 400),
            ("error.code context_length_exceeded", "context_length_exceeded",
             err.get("code"), err.get("code") == "context_length_exceeded"),
            ("message names the Engine max_context", "max_context 446902", msg[:120],
             "446902" in msg),
            ("no SSE opened (submit-time rejection)", "json body",
             "data:" not in txt[:200], "data:" not in txt[:200]),
        ]
        return {"checks": checks, "raw_head": txt[:800], "raw_tail": txt[-400:]}

    def probe_image():
        body = common_body(
            [{"role": "system", "content": "You are a vision-capable assistant."},
             {"role": "user", "content": [
                 {"type": "text", "text": "Describe this image in one short sentence."},
                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{TINY_PNG}"}},
             ]}],
            mct=128, reasoning_effort="xhigh")
        st, txt, _ = post_chat(body)
        events = parse_sse(txt) if st == 200 else []
        checks = [("image data URL with --vision unit -> 200", 200, st, st == 200)]
        if st == 200:
            checks += sse_contract(events)
        return {"checks": checks, "raw_head": txt[:1500], "raw_tail": txt[-1500:]}

    def probe_non_stream():
        body = common_body([{"role": "user", "content": "Reply with the single word: ok"}], mct=64)
        body["stream"] = False
        st, txt, _ = post_chat(body)
        j = {}
        try:
            j = json.loads(txt)
        except Exception:
            pass
        ch = (j.get("choices") or [{}])[0]
        checks = [
            ("stream false -> 200 aggregate", 200, st, st == 200),
            ("object == chat.completion", "chat.completion", j.get("object"),
             j.get("object") == "chat.completion"),
            ("finish_reason present", "yes", ch.get("finish_reason"),
             ch.get("finish_reason") in ("stop", "length", "tool_calls")),
            ("usage present", "yes", bool(j.get("usage")), bool(j.get("usage"))),
        ]
        return {"checks": checks, "raw_head": txt[:800], "raw_tail": txt[-400:]}

    GRAMMAR = '''root ::= "{" "color" ":" ( "red" | "blue" ) " intensity" ":" ( "low" | "high" ) "}"'''

    def probe_grammar():
        body = common_body(
            [{"role": "user", "content": "Emit the JSON object for the requested color: blue, high."}],
            mct=64, grammar=GRAMMAR)
        st, txt, _ = post_chat(body)
        checks = [("grammar request -> 200", 200, st, st == 200)]
        if st == 200:
            j = json.loads(txt)
            ch = (j.get("choices") or [{}])[0]
            content = (ch.get("message") or {}).get("content") or ""
            checks.append(("output satisfies the grammar", '{"color":"blue" "intensity":"high"}',
                           content, content.replace(" ", "") == '{"color":"blue" "intensity":"high"}'.replace(" ", "")
                           and json.loads(content) == {"color": "blue", "intensity": "high"}))
        else:
            checks.append(("body", "", txt[:300], False))
        return {"checks": checks, "raw_head": txt[:1000], "raw_tail": txt[-400:]}

    def probe_schema_json_object():
        body = common_body(
            [{"role": "user", "content": "Return a JSON object with a single field 'greeting' whose value is a short greeting."}],
            mct=128, response_format={"type": "json_object"})
        st, txt, _ = post_chat(body)
        checks = [("response_format json_object -> 200", 200, st, st == 200)]
        if st == 200:
            j = json.loads(txt)
            content = ((j.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            ok = False
            try:
                json.loads(content)
                ok = True
            except Exception:
                pass
            checks.append(("output is valid JSON", True, ok, ok))
        else:
            checks.append(("body", "", txt[:300], False))
        return {"checks": checks, "raw_head": txt[:1000], "raw_tail": txt[-400:]}

    def probe_schema_unsupported():
        body = common_body(
            [{"role": "user", "content": "hi"}], mct=64,
            response_format={"type": "json_schema", "json_schema": {
                "name": "x", "schema": {
                    "type": "object",
                    "properties": {"s": {"type": "string", "pattern": "^a$"}},
                }}})
        st, txt, _ = post_chat(body)
        j = {}
        try:
            j = json.loads(txt)
        except Exception:
            pass
        err = j.get("error", {})
        checks = [
            ("HTTP 400", 400, st, st == 400),
            ("error.code json_schema_unsupported naming pattern", "json_schema_unsupported",
             err.get("code"), err.get("code") == "json_schema_unsupported" and "pattern" in err.get("message", "")),
        ]
        return {"checks": checks, "raw_head": txt[:800], "raw_tail": txt[-400:]}

    def probe_constraint_conflict():
        body = common_body([{"role": "user", "content": "hi"}], mct=64,
                           grammar=GRAMMAR, response_format={"type": "json_object"})
        st, txt, _ = post_chat(body)
        j = {}
        try:
            j = json.loads(txt)
        except Exception:
            pass
        err = j.get("error", {})
        return {"checks": [
            ("HTTP 400", 400, st, st == 400),
            ("error.code constrained_decoding_conflict", "constrained_decoding_conflict",
             err.get("code"), err.get("code") == "constrained_decoding_conflict"),
        ], "raw_head": txt[:800], "raw_tail": txt[-400:]}

    def probe_constraint_with_tools():
        body = common_body([{"role": "user", "content": "hi"}], mct=64,
                           response_format={"type": "json_object"}, tools=[])
        st, txt, _ = post_chat(body)
        j = {}
        try:
            j = json.loads(txt)
        except Exception:
            pass
        err = j.get("error", {})
        return {"checks": [
            ("HTTP 400", 400, st, st == 400),
            ("error.code constrained_decoding_not_supported (constraint + tools [])",
             "constrained_decoding_not_supported", err.get("code"),
             err.get("code") == "constrained_decoding_not_supported"),
        ], "raw_head": txt[:800], "raw_tail": txt[-400:]}

    def probe_caddy():
        # the exact route opencode takes: https://ai.kido.ws/v1/chat/completions
        body = common_body(
            [{"role": "user", "content": "Reply with exactly: OK"}], mct=64)
        st, txt, hdrs = post_chat(body, base="https://ai.kido.ws/v1")
        events = parse_sse(txt) if st == 200 else []
        checks = [("via caddy (https://ai.kido.ws/v1) -> 200", 200, st, st == 200)]
        if st == 200:
            checks += sse_contract(events)
        return {"checks": checks, "raw_head": txt[:1500], "raw_tail": txt[-1500:]}

    # ---- run in order ----
    add("P01", "models", probe_models)
    add("P02", "health", probe_health)
    add("P03", "text_stream", probe_text_stream)
    add("P04", "max_precedence", probe_max_precedence)
    add("P05", "no_max_default", probe_no_max_default)
    add("P06", "tool_call", probe_tool_call)
    add("P07", "tool_replay", probe_tool_replay)
    add("P08", "tool_choice_none", probe_tool_choice_none)
    add("P09", "tool_choice_required_reject", probe_tool_choice_required)
    add("P10", "parallel_false_reject", probe_parallel_false)
    add("P11", "strict_true_reject", probe_strict_true)
    add("P12", "store_true_reject", probe_store_true)
    add("P13", "model_mismatch_404", probe_model_mismatch)
    add("P14", "unknown_field_tolerated", probe_unknown_field)
    add("P15", "effort_none", probe_effort_none)
    add("P16", "overflow_446902", probe_overflow)
    add("P17", "image_data_url", probe_image)
    add("P18", "non_stream", probe_non_stream)
    add("P19", "grammar_constrained", probe_grammar)
    add("P20", "schema_json_object", probe_schema_json_object)
    add("P21", "schema_unsupported_reject", probe_schema_unsupported)
    add("P22", "constraint_conflict_reject", probe_constraint_conflict)
    add("P23", "constraint_with_tools_reject", probe_constraint_with_tools)
    add("P24", "caddy_path", probe_caddy)

    with open(results_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    n_pass = sum(1 for r in recs if r["verdict"] == "pass")
    print(f"\n{n_pass}/{len(recs)} probes pass -> {results_path}")
    sys.exit(0 if n_pass == len(recs) else 1)


if __name__ == "__main__":
    main()
