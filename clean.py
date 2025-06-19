#!/usr/bin/env python3
"""
discord_export_to_dataset.py

Convert a Discord channel export (the JSON structure you posted) into
either an OpenAI-ready JSONL or a plain-text file for small GPTs.

Examples
--------
# OpenAI fine-tune
python discord_export_to_dataset.py channel.json dialog.jsonl --history 3 --fmt jsonl

# Andrej Karpathy nanoGPT-style corpus
python discord_export_to_dataset.py channel.json dialog.txt  --history 2 --fmt text
"""
import json
import argparse
from pathlib import Path

# Runs with `python3 clean.py DiscordLogs.json dialog.txt --history 0 --fmt text` using imported logs

# --------------------------------------------------------------------------- #
#                               ─── helpers ───                               #
# --------------------------------------------------------------------------- #
def load_messages(path: Path):
    """Return list of messages of `type == Default` (plain text posts)."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    msgs = data["messages"] if isinstance(data, dict) and "messages" in data else data
    return [m for m in msgs if m.get("type") == "Default" and m.get("content")]


def build_author_lookup(messages):
    """Map *message id* → display name (nickname if present, else username)."""
    lookup = {}
    for m in messages:
        author = m["author"]
        name = author.get("nickname") or author.get("name")
        lookup[m["id"]] = name
    return lookup


def format_line(msg, id_to_name):
    """
    Render one message as:
        <Alice>: hello there
    or, if it replies:
        <Bob>: [reply to Alice] sure!
    """
    author = id_to_name[msg["id"]]
    reply_hint = ""

    ref = msg.get("reference") or {}
    parent_id = ref.get("messageId")
    if parent_id and parent_id in id_to_name:
        reply_hint = f"[reply to {id_to_name[parent_id]}] "

    text = msg["content"].replace("\n", " ").strip()
    return f"<{author}>: {reply_hint}{text}"


def build_examples(messages, hist: int = 2):
    """
    Convert the raw message list into a list of
    {prompt: str, completion: str} dicts.

    `hist` = how many previous visible messages to prepend as context.
    """
    id_to_name = build_author_lookup(messages)
    examples = []

    for idx, msg in enumerate(messages):
        # previous `hist` messages become context
        ctx_msgs = messages[max(0, idx - hist): idx]
        prompt_lines = [format_line(m, id_to_name) for m in ctx_msgs]

        # prompt ends with *author tag only*; model must predict the text
        author_tag = f"<{id_to_name[msg['id']]}>:"
        prompt = "\n".join(prompt_lines + [author_tag])

        completion = " " + msg["content"].replace("\n", " ").strip()
        examples.append({"prompt": prompt, "completion": completion})

    return examples


def write_examples(examples, out_path: Path, fmt="jsonl"):
    """Serialize examples either as JSONL or plain text."""
    with out_path.open("w", encoding="utf-8") as f:
        if fmt == "jsonl":
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        elif fmt == "text":
            #  blank line between prompt & completion, *double* blank after pair
            for ex in examples:
                f.write(ex["prompt"].rstrip() + "\n")
                f.write(ex["completion"].lstrip() + "\n\n")
        else:
            raise ValueError("fmt must be 'jsonl' or 'text'")


# Optional wrapper to keep old call sites working --------------------------- #
def generate_jsonl(messages, history, outfile: Path):
    """Deprecated alias retained for backwards compatibility."""
    examples = build_examples(messages, hist=history)
    write_examples(examples, outfile, fmt="jsonl")


# --------------------------------------------------------------------------- #
#                                   CLI                                       #
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Discord export JSON file")
    p.add_argument("output", help="Destination file (jsonl or txt)")
    p.add_argument(
        "--history", "-n", type=int, default=2,
        help="How many previous messages to include as context")
    p.add_argument(
        "--fmt", choices=["jsonl", "text"], default="jsonl",
        help="Output format: jsonl (OpenAI) or text (char-level GPT)")

    args = p.parse_args()

    messages = load_messages(Path(args.input))
    examples = build_examples(messages, hist=args.history)
    write_examples(examples, Path(args.output), fmt=args.fmt)

    print(f"Wrote {len(examples):,} examples to {args.output} ({args.fmt.upper()})")


if __name__ == "__main__":
    main()
