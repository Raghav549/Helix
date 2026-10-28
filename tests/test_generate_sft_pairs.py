from scripts.generate_sft_pairs import generate_pairs


def test_sft_generation_is_deterministic_and_deduplicated():
    base = [["Answer briefly: What is 1 + 1?", "2"]]
    first, manifest = generate_pairs(base, seed=7, math_count=20, code_count=12, logic_count=10, conversation_count=10)
    second, _ = generate_pairs(base, seed=7, math_count=20, code_count=12, logic_count=10, conversation_count=10)
    assert first == second
    normalized = [(" ".join(a.split()).casefold(), " ".join(b.split()).casefold()) for a, b in first]
    assert len(normalized) == len(set(normalized))
    assert manifest["total_pairs"] == len(first)
    assert manifest["actual_counts"]["authored_base"] == 1
    assert manifest["actual_counts"]["math"] == 20
    assert manifest["actual_counts"]["logic"] == 10


def test_generated_arithmetic_answers_are_exact():
    pairs, _ = generate_pairs([], seed=17, math_count=80, code_count=0, logic_count=0, conversation_count=0)
    for instruction, response in pairs:
        if "Calculate exactly:" in instruction:
            expression = instruction.split("Calculate exactly: ", 1)[1].split(". Return", 1)[0]
            a, op, b = expression.split()
            a, b = int(a), int(b)
            expected = {"+": a + b, "-": a - b, "*": a * b, "/": a // b}[op]
            assert int(response) == expected


def test_negative_generation_counts_are_rejected():
    import pytest
    with pytest.raises(ValueError):
        generate_pairs([], math_count=-1)


def test_requested_code_logic_and_conversation_counts_produce_distinct_pairs():
    _, manifest = generate_pairs([], seed=23, math_count=0, code_count=60, logic_count=80, conversation_count=40)
    counts = manifest["actual_counts"]
    assert counts["code"] >= 50
    assert counts["logic"] >= 70
    assert counts["conversation"] == 40
