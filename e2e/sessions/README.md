# Real opencode sessions (ticket #143)

Driven 2026-10-03 ~00:01-00:08Z as:

    opencode run --standalone --model "ai.kido.ws/Qwen3.8-27B#xhigh" --title e2e143-wN --auto --format json "<task>"

(opencode v2.0.22). Per dir: `out.json` (captured JSON event stream), `err.log`,
`meta.json` (exit code, wall clock, session id, task prompt). W2 additionally has
the serve-side and opencode-store evidence quoted in the findings doc (session
`ses_f0180baaeffe2IOtQKxoOfHD5X`; the full 100 KB assistant message was read from
`~/.local/share/opencode/opencode.db` table `session_message`, not copied here).
