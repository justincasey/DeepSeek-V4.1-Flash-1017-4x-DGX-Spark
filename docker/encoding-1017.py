# SPDX-License-Identifier: Apache-2.0
"""Install DeepSeek's DeepSeek-V4.1-Flash-1017 prompt encoder into vLLM. Runs once at image build.

The base image's vllm/tokenizers/deepseek_v41_encoding.py implements the pre-1017 grammar. 1017 changed it:
reasoning-effort tiers low/high/max -> 50/75/100 (default 75; was 25/50/75/100 with xhigh, default 50),
tool namespaces, tool results carrying content blocks and images, tool-result-only user turns no longer starting
a new turn, and image-placeholder validation for tool messages.

This step replaces the module with the publisher's encoding/encoding.py (MIT, DeepSeek) fetched from the pinned
checkpoint revision and checked against the recipe's sha256 manifest, then adapts vLLM's single caller
(vllm/tokenizers/deepseek_v41.py):
  - content lists are handed to the encoder as content blocks (the 1017 encoder places image tokens itself and
    rejects placeholder text);
  - every reasoning_effort value the OpenAI schema accepts renders: minimal -> low, medium -> high, xhigh -> max.
The build fails unless the publisher's golden fixtures encode byte-for-byte.
"""
import argparse
import hashlib
import importlib.util
import os
import sys
import tempfile
import urllib.request

VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"
ENCODER = f"{VLLM}/tokenizers/deepseek_v41_encoding.py"
WRAPPER = f"{VLLM}/tokenizers/deepseek_v41.py"


def fetch(repo, revision, path, manifest):
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{path}"
    data = urllib.request.urlopen(url, timeout=120).read()
    digest = hashlib.sha256(data).hexdigest()
    if manifest.get(path) != digest:
        sys.exit(f"{path}: sha256 {digest} does not match the manifest ({manifest.get(path)})")
    return data


def replace_once(text, old, new, what):
    if text.count(old) != 1:
        sys.exit(f"anchor for {what} found {text.count(old)} times in {WRAPPER}")
    return text.replace(old, new)


NORMALIZE_OLD = '''        if isinstance(content, list):
            parts = []
            for block in content:
                part_type = block.get("type")
                if part_type == "text":
                    parts.append(block.get("text", ""))
                elif part_type in ("image_url", "input_image", "image_pil"):
                    parts.append(IMAGE_PLACEHOLDER)
                else:
                    raise ValueError(
                        "DeepSeek V4.1 supports text and image content only; "
                        f"got {part_type!r}"
                    )
            message["content"] = "\\n\\n".join(parts)
'''
NORMALIZE_NEW = '''        if isinstance(content, list):
            # dsv41-1017: hand content blocks to the 1017 encoder, which joins text with
            # blank lines and places the image token itself; vLLM's multimodal processor
            # supplies the pixels in the same order.
            blocks = []
            for block in content:
                part_type = block.get("type")
                if part_type == "text":
                    blocks.append({"type": "text", "text": block.get("text", "")})
                elif part_type in ("image_url", "input_image", "image_pil", "image"):
                    blocks.append({"type": "image_url", "image_url": {"url": "vllm:image"}})
                else:
                    raise ValueError(
                        "DeepSeek V4.1 supports text and image content only; "
                        f"got {part_type!r}"
                    )
            message["content"] = blocks
'''
EFFORT_OLD = '''            if effort is None:
                effort = "high"
'''
EFFORT_NEW = '''            if effort is None:
                effort = "high"
            # dsv41-1017: the API schema also accepts OpenAI's minimal/medium/xhigh;
            # map them onto the nearest 1017 tier (low/high/max = 50/75/100).
            effort = {"minimal": "low", "medium": "high", "xhigh": "max"}.get(effort, effort)
'''
MESSAGE_OLD = '"DeepSeek V4.1 reasoning_effort must be low, high, xhigh, max, "'
MESSAGE_NEW = '"DeepSeek V4.1 reasoning_effort must be none, minimal, low, medium, high, xhigh, max, "'


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--manifest", required=True)
    a = p.parse_args()
    manifest = {}
    for line in open(a.manifest):
        if line.strip():
            digest, path = line.split(None, 1)
            manifest[path.strip()] = digest

    encoder = fetch(a.repo, a.revision, "encoding/encoding.py", manifest)
    header = (f"# DeepSeek-V4.1-Flash-1017 reference encoder, {a.repo}@{a.revision} encoding/encoding.py\n"
              "# Copyright (c) 2026 DeepSeek. MIT License. Installed unmodified by the dsv41-1017 recipe.\n").encode()
    with open(ENCODER, "wb") as f:
        f.write(header + encoder)

    w = open(WRAPPER).read()
    if "dsv41-1017" not in w:
        w = replace_once(w, NORMALIZE_OLD, NORMALIZE_NEW, "content normalization")
        w = replace_once(w, EFFORT_OLD, EFFORT_NEW, "effort aliases")
        w = replace_once(w, MESSAGE_OLD, MESSAGE_NEW, "effort error message")
        compile(w, WRAPPER, "exec")
        open(WRAPPER, "w").write(w)

    # Golden fixtures shipped with the encoder must render byte-for-byte.
    spec = importlib.util.spec_from_file_location("enc1017", ENCODER)
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    assert enc.REASONING_EFFORT_MAPPINGS == {"low": 50, "high": 75, "max": 100}, enc.REASONING_EFFORT_MAPPINGS
    cases = sorted(p for p in manifest if p.startswith("encoding/tests/test_input_"))
    with tempfile.TemporaryDirectory() as tmp:
        for path in cases:
            out_path = path.replace("test_input_", "test_output_").replace(".json", ".txt")
            local = os.path.join(tmp, os.path.basename(path))
            open(local, "wb").write(fetch(a.repo, a.revision, path, manifest))
            want = fetch(a.repo, a.revision, out_path, manifest).decode()
            got, _ = enc.encode_case(enc.load_cases(local)[0], thinking_mode="chat")
            if got != want:
                sys.exit(f"golden mismatch for {path}")
    print(f"1017 encoder installed; {len(cases)} golden fixtures match; wrapper adapted")


if __name__ == "__main__":
    main()
