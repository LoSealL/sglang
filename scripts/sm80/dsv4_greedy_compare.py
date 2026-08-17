#!/usr/bin/env python
"""Greedy cross-check of the DSV4-Flash sm80 port: sglang vs vLLM.

Two modes (servers run sequentially; they cannot share GPUs):

  gen: hit one server's /v1/completions with the fixed prompts below
       (temperature=0, max_tokens=128, ignore_eos) and dump a JSON list of
       {prompt, text}.
  cmp: load two dumps, report per-prompt common text prefix and token-level
       common greedy prefix (via the HF tokenizer of --model).

Usage:
  python dsv4_greedy_compare.py gen --url http://localhost:8001 --out /tmp/opencode/vllm_refs.json
  python dsv4_greedy_compare.py gen --url http://localhost:8000 --out /tmp/opencode/sgl_out.json
  python dsv4_greedy_compare.py cmp --ref /tmp/opencode/vllm_refs.json \
      --out /tmp/opencode/sgl_out.json --model $MODEL

Acceptance: >=4/5 prompts with >=64 identical leading tokens.
"""

import argparse
import json
import urllib.request

PROMPTS = [
    "The capital of France is",
    "Sarah has 3 boxes with 12 pencils in each box. She gives 7 pencils to "
    "her friend and then buys 2 more boxes with 12 pencils each. How many "
    "pencils does Sarah have now?",
    "Write a Python function that computes the n-th Fibonacci number "
    "iteratively:",
    "List the first five planets of the solar system in order from the Sun:",
    "The lighthouse keeper had tended the beam for thirty years. Every night "
    "at dusk he climbed the two hundred steps, lit the lamp, and watched the "
    "fog roll in from the sea. One evening, as the autumn storms began, he "
    "noticed a small boat struggling against the waves near the rocks. He "
    "continued the story:",
]

MAX_TOKENS = 128


def gen(args: argparse.Namespace) -> None:
    results = []
    for i, prompt in enumerate(PROMPTS):
        body = json.dumps(
            {
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": MAX_TOKENS,
                "ignore_eos": True,
            }
        ).encode()
        req = urllib.request.Request(
            args.url + "/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.load(r)
        text = data["choices"][0]["text"]
        results.append({"prompt": prompt, "text": text})
        print(f"[{i + 1}/{len(PROMPTS)}] {len(text)} chars")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {args.out}")


def common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def cmp_mode(args: argparse.Namespace) -> None:
    with open(args.ref) as f:
        ref = json.load(f)
    with open(args.out) as f:
        out = json.load(f)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    n_ok = 0
    for i, (r, o) in enumerate(zip(ref, out)):
        chars = common_prefix_len(r["text"], o["text"])
        # ponytail: encode full texts independently; BPE only diverges at/after
        # the text divergence, so the leading equal ids are a faithful measure
        rt = tok.encode(r["text"])
        ot = tok.encode(o["text"])
        toks = 0
        for x, y in zip(rt, ot):
            if x != y:
                break
            toks += 1
        n_ok += toks >= 64
        print(
            f"prompt {i}: common prefix {toks} tokens / {chars} chars "
            f"(lens {len(rt)} vs {len(ot)} tokens)"
        )
    verdict = "PASS" if n_ok >= 4 else "FAIL"
    print(f"{n_ok}/5 prompts with >=64 identical greedy tokens -> {verdict}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--url", required=True)
    g.add_argument("--out", required=True)
    c = sub.add_parser("cmp")
    c.add_argument("--ref", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--model", required=True)
    args = p.parse_args()
    {"gen": gen, "cmp": cmp_mode}[args.mode](args)


if __name__ == "__main__":
    main()
