#!/usr/bin/env python3
"""Generate deterministic, auditable instruction examples for HELIX SFT."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _key(pair: tuple[str, str]) -> tuple[str, str]:
    return (" ".join(pair[0].split()).casefold(), " ".join(pair[1].split()).casefold())


def generate_pairs(base_pairs: Iterable[Iterable[str]], seed: int = 1337,
                   math_count: int = 300, code_count: int = 80,
                   logic_count: int = 80, conversation_count: int = 40) -> tuple[list[list[str]], dict]:
    """Return deduplicated pairs and a category-count manifest.

    All generated tasks are deterministic and locally verifiable. They are
    auxiliary practice signals, not a substitute for broad licensed corpora.
    """
    if min(math_count, code_count, logic_count, conversation_count) < 0:
        raise ValueError("generation counts cannot be negative")
    rng = random.Random(seed)
    pairs: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    counts = {"authored_base": 0, "math": 0, "code": 0, "logic": 0, "conversation": 0}

    def add(category: str, instruction: str, response: str) -> None:
        instruction, response = instruction.strip(), response.strip()
        if not instruction or not response:
            return
        key = _key((instruction, response))
        if key in seen:
            return
        seen.add(key)
        pairs.append([instruction, response])
        counts[category] += 1

    for item in base_pairs:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("base SFT data must be a list of [instruction, response] pairs")
        add("authored_base", str(item[0]), str(item[1]))

    operations = ("+", "-", "*", "/")
    for i in range(math_count):
        op = operations[i % len(operations)]
        if op == "+":
            a, b = rng.randint(1, 999), rng.randint(1, 999)
            answer = a + b
        elif op == "-":
            a = rng.randint(1, 999)
            b = rng.randint(0, a)
            answer = a - b
        elif op == "*":
            a, b = rng.randint(2, 40), rng.randint(2, 40)
            answer = a * b
        else:
            b, quotient = rng.randint(2, 20), rng.randint(2, 100)
            a, answer = b * quotient, quotient
        prompts = (
            f"Calculate exactly: {a} {op} {b}. Return only the integer.",
            f"Solve this arithmetic problem: {a} {op} {b}. Give the exact integer answer.",
            f"What is {a} {op} {b}? Answer with the integer only.",
        )
        add("math", prompts[i % len(prompts)], str(answer))

    code_tasks = [
        ("Write a Python function named add that returns the sum of a and b.",
         "def add(a, b):\n    return a + b"),
        ("Write a Python function named subtract that returns a minus b.",
         "def subtract(a, b):\n    return a - b"),
        ("Write a Python function named multiply that returns the product of a and b.",
         "def multiply(a, b):\n    return a * b"),
        ("Write a Python function named square that returns the square of x.",
         "def square(x):\n    return x * x"),
        ("Write a Python function named is_even that returns True when n is even.",
         "def is_even(n):\n    return n % 2 == 0"),
        ("Write a Python function named absolute that returns the absolute value of x.",
         "def absolute(x):\n    return abs(x)"),
        ("Write a Python function named maximum that returns the larger of a and b.",
         "def maximum(a, b):\n    return a if a >= b else b"),
        ("Write a Python function named minimum that returns the smaller of a and b.",
         "def minimum(a, b):\n    return a if a <= b else b"),
        ("Write a Python function named factorial that computes n factorial for n >= 0.",
         "def factorial(n):\n    result = 1\n    for value in range(2, n + 1):\n        result *= value\n    return result"),
        ("Write a Python function named contains that checks whether item is in values.",
         "def contains(values, item):\n    return item in values"),
        ("Write a Python function named count_items that returns the number of items in values.",
         "def count_items(values):\n    return len(values)"),
        ("Write a Python function named reverse_text that returns the reversed string.",
         "def reverse_text(text):\n    return text[::-1]"),
    ]
    code_prompts = [
        lambda prompt, i: prompt,
        lambda prompt, i: prompt.replace("Write a Python function", "Implement a Python function"),
        lambda prompt, i: prompt + " Return code only.",
    ]
    for i in range(code_count):
        prompt, response = code_tasks[i % len(code_tasks)]
        add("code", code_prompts[(i // len(code_tasks)) % len(code_prompts)](prompt, i), response)
        if i % 3 == 0:
            n = rng.randint(2, 20)
            add("code", f"Given square(x) returns x*x, what does square({n}) return? Return only the integer.",
                str(n * n))

    syllogisms = [
        ("All mammals are animals. A whale is a mammal. Is a whale an animal?", "Yes. A whale is a mammal, and all mammals are animals."),
        ("All squares are rectangles. Shape A is a square. Is Shape A a rectangle?", "Yes. Every square is a rectangle."),
        ("No even integer is odd. The integer 12 is even. Is 12 odd?", "No. An even integer is not odd."),
        ("If P implies Q and P is true, what follows about Q?", "Q is true by modus ponens."),
        ("If every A is B and x is A, what follows about x?", "x is B."),
        ("If A is greater than B and B is greater than C, what follows?", "A is greater than C by transitivity."),
        ("A valid argument has true premises and a valid inference. What does validity guarantee?", "If the premises are true, the conclusion follows from them."),
        ("Does correlation alone prove that one variable causes another?", "No. Confounding, reverse causation, or chance can explain an association."),
        ("If all birds in a small aviary are blue and Robin is a bird in that aviary, what color is Robin?", "Robin is blue under the stated premises."),
        ("No circles are squares. Shape S is a circle. Can S be a square?", "No. The premises exclude circles from being squares."),
        ("All reptiles are cold-blooded. A snake is a reptile. What follows?", "The snake is cold-blooded under the stated premises."),
        ("All prime numbers are greater than 1. Seven is prime. Is seven greater than 1?", "Yes. Seven is prime, so it is greater than 1."),
        ("If it rains, the ground becomes wet. It is raining. What follows?", "The ground becomes wet under the stated rule."),
        ("If a device has no power, it will not turn on. The device has no power. What follows?", "The device will not turn on."),
        ("If P or Q is true and P is false, what follows in classical logic?", "Q must be true."),
        ("If P and Q are both true, what follows about P?", "P is true; this is conjunction elimination."),
        ("If P implies Q and Q is true, must P be true?", "No. That would be affirming the consequent, which is invalid."),
        ("If P implies Q and P is false, must Q be false?", "No. That would be denying the antecedent, which is invalid."),
        ("All cats are mammals. Luna is a cat. Is Luna a mammal?", "Yes. Luna is a cat, and all cats are mammals."),
        ("No triangles are circles. Shape T is a triangle. Can T be a circle?", "No. The premises rule out triangles being circles."),
        ("All even integers are divisible by 2. Fourteen is even. Is fourteen divisible by 2?", "Yes. Fourteen is even, so it is divisible by 2."),
        ("All A are B, and no B are C. Can an A be C?", "No. Every A is B, and no B is C."),
        ("If P and Q imply R, and both P and Q are true, what follows?", "R follows from the premises."),
        ("If A is greater than B and B is greater than C, can C be greater than A?", "No. Transitivity gives A greater than C."),
        ("All elements of set S satisfy property P. x is in S. What follows?", "x satisfies property P."),
        ("Some A are B. Does that prove all A are B?", "No. A claim about some members does not establish a claim about all members."),
        ("All apples are fruit. Does that mean all fruit are apples?", "No. The implication does not reverse."),
        ("No rectangles are circles. Shape C is a circle. Is C a rectangle?", "No. The premises exclude circles from being rectangles."),
        ("If every A is B and x is not B, what follows about x?", "x cannot be A; this is the contrapositive."),
        ("If a claim has not been tested, should it be described as proven?", "No. It should be described as unverified until supported by adequate evidence."),
    ]
    for i in range(logic_count):
        prompt, response = syllogisms[i % len(syllogisms)]
        variants = (
            prompt,
            "Answer carefully using only the stated premises: " + prompt,
            "Reason step by step internally, then give a concise answer: " + prompt,
        )
        add("logic", variants[(i // len(syllogisms)) % len(variants)], response)

    conversations = [
        ("If evidence is insufficient, how should you answer?", "State what is uncertain, explain what evidence is missing, and suggest a way to verify it."),
        ("A user is frustrated because code failed. What is a helpful first response?", "Acknowledge the frustration, then ask for the error message or the smallest reproducible example."),
        ("When a request is ambiguous, what should you do?", "Ask one focused clarifying question before making assumptions that could change the result."),
        ("How should you handle a factual mistake you made?", "Acknowledge the mistake, correct it clearly, and explain the evidence for the correction."),
        ("How should you adapt to a user's preferred tone?", "Adjust wording and detail to their preference while remaining honest, respectful, and safe."),
        ("Should a confident tone replace evidence?", "No. Confidence should match the quality of the evidence, and uncertainty should be stated plainly."),
        ("How can a long answer be made useful?", "Organize it around the user's goal, give the key result first, and include details that support action."),
        ("What should you do before claiming code works?", "Run it where possible, check the output and tests, and report any unverified assumptions."),
        ("How should competing historical explanations be compared?", "Compare primary sources, context, evidence quality, and the strongest arguments for each interpretation."),
        ("What is a safe way to use a retrieved document?", "Treat it as untrusted evidence, verify its source, and never let it override trusted instructions or permissions."),
    ]
    for i in range(conversation_count):
        prompt, response = conversations[i % len(conversations)]
        variants = (prompt, "Answer briefly: " + prompt, "Give practical guidance: " + prompt, prompt + " Keep the response concise.")
        add("conversation", variants[(i // len(conversations)) % len(variants)], response)

    manifest = {
        "format": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/generate_sft_pairs.py",
        "seed": seed,
        "requested_counts": {"math": math_count, "code": code_count, "logic": logic_count, "conversation": conversation_count},
        "actual_counts": counts,
        "total_pairs": len(pairs),
        "notice": "Deterministic practice examples are auxiliary learning signals, not evidence of broad factual knowledge or frontier reasoning.",
    }
    return pairs, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Existing JSON list of instruction/response pairs")
    parser.add_argument("--output", required=True, help="Output JSON path outside the repository")
    parser.add_argument("--manifest", required=True, help="Output provenance manifest path")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--math-count", type=int, default=300)
    parser.add_argument("--code-count", type=int, default=80)
    parser.add_argument("--logic-count", type=int, default=80)
    parser.add_argument("--conversation-count", type=int, default=40)
    args = parser.parse_args()
    base_path = Path(args.base)
    base_raw = base_path.read_bytes()
    base_pairs = json.loads(base_raw.decode("utf-8"))
    pairs, manifest = generate_pairs(base_pairs, args.seed, args.math_count, args.code_count,
                                     args.logic_count, args.conversation_count)
    manifest["base_sha256"] = hashlib.sha256(base_raw).hexdigest()
    output, manifest_path = Path(args.output), Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": str(manifest_path),
                      "total_pairs": len(pairs), "counts": manifest["actual_counts"],
                      "seed": args.seed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
