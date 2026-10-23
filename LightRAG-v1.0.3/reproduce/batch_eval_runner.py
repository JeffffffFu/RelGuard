"""Pairwise LLM-judge evaluation, faithful to the LightRAG paper (arXiv 2410.05779).

Replaces examples/batch_eval.py, which cannot run: it targets the OpenAI Batch
API, calls batch_eval() with no arguments, drops "Diversity" from the requested
JSON schema (a dimension Table 1 reports), and never fetches results or
aggregates win rates.

This runner follows the paper instead:
  * judge prompt from Figure 7 -- all four keys, Diversity included
  * GPT-4o-mini as the judge (paper, section 4.1)
  * "we alternate the placement of each answer in the prompt" -- every pair is
    judged twice with the two answers swapped, so position bias is measured
    rather than assumed away.

Usage:
    python batch_eval_runner.py <system_a_name> <result_a.json> \
                               <system_b_name> <result_b.json> [out.json]
"""

from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
import re
import sys
from collections import defaultdict

from openai import AsyncOpenAI

JUDGE_MODEL = "gpt-4o-mini"
MAX_CONCURRENCY = 8
DIMENSIONS = ["Comprehensiveness", "Diversity", "Empowerment", "Overall Winner"]

SYS_PROMPT = """
---Role---
You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
"""

USER_PROMPT = """
You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

Here is the question:
{query}

Here are the two answers:

**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:

{{
    "Comprehensiveness": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Diversity": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Overall Winner": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
    }}
}}
"""

client = AsyncOpenAI()


def _slot(winner: str) -> int | None:
    if "1" in winner:
        return 1
    if "2" in winner:
        return 2
    return None


def parse_verdict(raw: str) -> dict[str, int] | None:
    """Pull the four winners out of the judge's reply.

    Strict JSON first. It fails often enough to matter: the judge writes the
    quoted titles it finds in the answers straight into the Explanation strings
    without escaping them, which breaks the object. So fall back to reading just
    the Winner fields, which are short and never carry embedded quotes -- the
    explanations are not used for scoring anyway.
    """
    block = re.search(r"\{.*\}", raw, re.S)
    if block:
        try:
            data = json.loads(block.group(0))
            out = {}
            for dim in DIMENSIONS:
                slot = _slot(str((data.get(dim) or {}).get("Winner", "")))
                if slot is None:
                    break
                out[dim] = slot
            if len(out) == len(DIMENSIONS):
                return out
        except json.JSONDecodeError:
            pass

    out = {}
    for dim in DIMENSIONS:
        m = re.search(
            rf'"{re.escape(dim)}"\s*:\s*\{{[^}}]*?"Winner"\s*:\s*"([^"]*)"', raw, re.S
        )
        if not m:
            return None
        slot = _slot(m.group(1))
        if slot is None:
            return None
        out[dim] = slot
    return out


async def judge_once(sem, query, answer1, answer2):
    async with sem:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": SYS_PROMPT},
                        {
                            "role": "user",
                            "content": USER_PROMPT.format(
                                query=query, answer1=answer1, answer2=answer2
                            ),
                        },
                    ],
                )
                verdict = parse_verdict(r.choices[0].message.content or "")
                if verdict:
                    return verdict
            except Exception as e:  # rate limit, transient 5xx, malformed JSON
                if attempt == 2:
                    print(f"  judge failed: {type(e).__name__}: {str(e)[:80]}")
                await asyncio.sleep(2 * (attempt + 1))
        return None


async def main(name_a, file_a, name_b, file_b, out_path):
    queries = [x["query"] for x in json.load(open(file_a))]
    ans_a = {x["query"]: x["result"] for x in json.load(open(file_a))}
    ans_b = {x["query"]: x["result"] for x in json.load(open(file_b))}
    shared = [q for q in queries if q in ans_b]
    print(f"{name_a}: {len(ans_a)} answers | {name_b}: {len(ans_b)} answers")
    print(f"evaluating {len(shared)} shared questions x 2 orderings "
          f"= {2 * len(shared)} judge calls\n")

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = []
    for q in shared:
        # ordering 1: A is "Answer 1";  ordering 2: A is "Answer 2"
        tasks.append(judge_once(sem, q, ans_a[q], ans_b[q]))
        tasks.append(judge_once(sem, q, ans_b[q], ans_a[q]))
    results = await asyncio.gather(*tasks)

    # wins[dim][system] counted over both orderings
    wins = defaultdict(lambda: defaultdict(int))
    pos_wins = defaultdict(int)  # how often slot "Answer 1" won, any system
    valid = 0
    per_question = []

    for i, q in enumerate(shared):
        fwd, rev = results[2 * i], results[2 * i + 1]
        rec = {"query": q}
        for dim in DIMENSIONS:
            for verdict, a_is_slot1 in ((fwd, True), (rev, False)):
                if verdict is None:
                    continue
                slot = verdict[dim]
                winner = name_a if (slot == 1) == a_is_slot1 else name_b
                wins[dim][winner] += 1
                pos_wins[dim] += 1 if slot == 1 else 0
            rec[dim] = {
                "forward": fwd[dim] if fwd else None,
                "reverse": rev[dim] if rev else None,
            }
        per_question.append(rec)
        valid += (fwd is not None) + (rev is not None)

    print(f"valid judge responses: {valid}/{2 * len(shared)}\n")
    print(f"{'Dimension':<20} {name_a:>14} {name_b:>14}")
    print("-" * 50)
    table = {}
    for dim in DIMENSIONS:
        a, b = wins[dim][name_a], wins[dim][name_b]
        tot = a + b
        if not tot:
            continue
        table[dim] = {name_a: 100 * a / tot, name_b: 100 * b / tot, "n": tot}
        print(f"{dim:<20} {100 * a / tot:>13.1f}% {100 * b / tot:>13.1f}%")

    print("\nposition bias (how often the slot 'Answer 1' won, 50% = unbiased):")
    for dim in DIMENSIONS:
        tot = wins[dim][name_a] + wins[dim][name_b]
        if tot:
            print(f"  {dim:<20} {100 * pos_wins[dim] / tot:>5.1f}%")

    json.dump(
        {"system_a": name_a, "system_b": name_b, "win_rates": table,
         "per_question": per_question},
        open(out_path, "w"), indent=2, ensure_ascii=False,
    )
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)
    a_name, a_file, b_name, b_file = sys.argv[1:5]
    out = sys.argv[5] if len(sys.argv) > 5 else "eval_result.json"
    asyncio.run(main(a_name, a_file, b_name, b_file, out))
