import asyncio
import json
import os
import re
import time
from typing import Optional

from recipe.usim.utils import extract_json, parse_messages

HIERARCHY_PROMPT_BATCHED = '''You are a helpful and meticulous evaluator. \
Your task is to score how well the generated {hierarchy_name}(s) align with the ground truth user response. \
Description of {hierarchy_name}: {hierarchy_desc}.

You will be given the context, the ground truth response, and generated {hierarchy_name}(s) that you should evaluate.

Provided Information:
<|The Start of Context|>
{context}
<|The End of Context|>

<|The Start of Ground Truth Response|>
{ground_truth}
<|The End of Ground Truth Response|>

{generations_text}

CRITICAL PRE-CHECK - Speaker Perspective Check (MUST DO FIRST):
Before scoring content similarity, verify that the generation is describing the HUMAN's {hierarchy_name}, not the assistant's. This is a gating check - if the generation is from the wrong perspective, it automatically fails.

The task is to capture the HUMAN's (user's) {hierarchy_name}. The generation should describe what the HUMAN expressed, NOT what the assistant said or would say.

Signs the generation is incorrectly capturing the ASSISTANT's perspective (automatic score = 0):
- Describes helping, explaining, assisting, or responding to the user
- Uses framing like "provide information about...", "help the user with...", "address the user's concern about..."
- Captures a VERY neutral, balanced, or hedged perspective typical of assistant responses
- Describes the assistant's intent or behavior rather than the HUMAN's

Examples of assistant-perspective errors:
- HUMAN asks for debugging help. Wrong generation for GOAL: "help the user debug their code" (assistant's goal). Correct: "get help fixing a bug" (HUMAN's goal).
- HUMAN expresses strong opinion on remote work. Wrong generation for STANCE: "remote work has advantages and disadvantages" (assistant's neutral framing). Correct: "strongly favors remote work" (HUMAN's stance).
- HUMAN asks "Can you explain how photosynthesis works?" Wrong generation for RESPONSE: "Photosynthesis is the process by which plants convert sunlight into energy..." (assistant's explanation). Correct: the question itself "Can you explain how photosynthesis works?"
- HUMAN says "squeeze this text to 200 words: [long text]" Wrong generation for RESPONSE: a 200-word summary of the text (assistant completing the task). Correct: the request itself "squeeze this text to 200 words: [long text]"

Note: Sometimes the HUMAN and assistant may genuinely share similar {hierarchy_name}s - this is fine. The check is specifically to catch when the generation describes the assistant's perspective INSTEAD OF the HUMAN's.

If the generation is from the assistant's perspective, assign score = 0 and skip remaining steps.

Scoring Criteria (only if pre-check passes):
For each generated {hierarchy_name}, assign a score in [0, 1] based on how accurately it reflects the ground truth response.

Guidelines:
1. Extract 1-3 key points:
   - Extract K key points from the ground truth response along the {hierarchy_name} dimension (e.g., if evaluating a "stance", pick key points related to the stance like "clearly disagrees with X", if evaluating a "response", pick key points about the response like "offers a solution to Y").
   - If {hierarchy_name} is different from "a response" (e.g., "stance", "target"), focus on key points only relevant to the {hierarchy_name} of the response.
   - Each key point should be specific and distinct.

2. Score how well the generated {hierarchy_name} matches each key point:
   - For each key point i, compare it with the generated {hierarchy_name} and assign a match value m_i in range [0, 1]:
     - 1.0: The key point is precisely and perfectly reflected.
     - [0.7, 0.9]: Mostly reflected with small imperfections.
     - [0.4, 0.6]: Partially reflected or vague, but still leaning in the correct direction.
     - [0.1, 0.3]: Very weak reflection.
     - 0.0: Missed, contradicted, or reversed.

3. Compute coverage C = (m_1 + m_2 + ... + m_K) / K, which measures how comprehensive the generated {hierarchy_name} reflects the ground truth response.

4. Compute penalty P for extra or conflicting content:
   - Examine additional content in the generated {hierarchy_name} beyond those key points:
     - Does it introduce unsupported evidence and assumptions?
     - Is it irrelevant to what ground truth response expresses?
   - Set a penalty P ∈ [0, 1]:
     - 0.0: No problematic extra content; everything is perfectly matched.
     - [0.1, 0.3]: Slightly unnecessary or mildly speculative detail; meaning essentially unchanged.
     - [0.4, 0.6]: Moderate speculative or irrelevant content that somewhat shifts emphasis or adds unsupported ideas.
     - [0.7, 0.9]: Significant speculative, misleading, or conflicting content that clearly changes the meaning.
     - 1.0: Mostly off-topic, contradictory, or dominated by incorrect/hallucinated content.

5. If you are evaluating generated responses (skip if {hierarchy_name} is not a response):
   - Length alone does NOT increase the score. Extra length is only ok if it is consistent and not redundant.
   - A generated response that is much longer than the ground truth response should be penalized via P.
   - The generated response may or may not reuse phrases from the context; however, if the generated response just directly copies previous context, without quoting them, treat that as off-task behavior and give a score of 0.

6. Compute the final score = max(0, min(1, C - P))

Additional considerations:
- Follow the instruction carefully.
- Be strict and reserve scores above 0.8 for clearly outstanding matches.
{other_guidelines}

Output format (JSON):
{{
    "key_points": "<analysis of key points from ground truth along {hierarchy_name} dimension>",
    "1": {{"thought": "<how well the 1st generated {hierarchy_name} matches each key point and compute the final score>", "score": <score>}},
    "2": ...
}}

Format Notes:
- All text in "key_points" and "thought" fields MUST be on a single line with no line breaks or newlines
- Use standard JSON string format with double quotes. For any quotes needed inside strings, use single quotes (')
- Double check the JSON array's format, especially for the comma and quotation marks
- Ensure that ALL fields, especially "thought" and "score", are present for each item
- You must provide exactly {num_generations} scores for the generated {hierarchy_name}(s)

Your output:
'''


def extract_usage(resp):
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")

    # Convert pydantic-ish objects to dict when possible
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif hasattr(usage, "dict"):
        usage = usage.dict()

    return usage or {}


INVALID_NAME_CHARS_RE = re.compile(r'[\s<|\\/>]')



async def compute_batch_score(data_source, generations, ground_truth, extra_info, **kwargs) -> list[float]:
    """
    Batched version: scores multiple generations at once.
    Returns a list of floats, one score per generation.
    """
    if not generations:
        return []

    
    max_retry = kwargs.pop("max_retry", 5)
    num_repeats = kwargs.pop("num_repeats", 1)
    hierarchy_name = extra_info['hierarchy_name']
    hierarchy_desc = extra_info["hierarchy_desc"]

    raw_prompt = json.loads(extra_info["raw_prompt"])
    context = parse_messages(raw_prompt)
    
    # Format all generations as a dict
    generations_dict = {i+1: gen.strip() for i, gen in enumerate(generations)}
    generations_text = f"<|The Start of Generated {hierarchy_name}s|>\n{json.dumps(generations_dict, indent=2)}\n<|The End of Generated {hierarchy_name}s|>"
    
    other_guidelines = f"- If a {hierarchy_name} contains non-text content, unnecessary wrappers like XML-like markup, or is otherwise malformed, apply a penalty by multiplying its score by 0.5. If there are multiple {hierarchy_name}s, you should contrast them against each other to ensure that your evaluations are consistent and assign different scores to different generated {hierarchy_name}s."
    
    # overwrite
    if 'other_guidelines' in kwargs:
        other_guidelines = kwargs.pop('other_guidelines')

    prompt = HIERARCHY_PROMPT_BATCHED.format(
        context=context,
        ground_truth=ground_truth,
        generations_text=generations_text,
        other_guidelines=other_guidelines,
        hierarchy_name=hierarchy_name,
        hierarchy_desc=hierarchy_desc,
        num_generations=len(generations)
    )


    
    model = kwargs.pop("model", "openai/gpt-5-mini")
    # slight randomness to avoid getting stuck in a loop
    temperature = kwargs.pop("temperature", 1)
    max_tokens = kwargs.pop("max_tokens", 4096)

    for repeat in range(num_repeats):
        scores_list = []
        for attempt in range(max_retry):
            content = None
            
            # Try litellm first
            try:
                import litellm
                resp = await litellm.acompletion(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                content = resp.choices[0].message.content
            except Exception as e:
                print(f"[Attempt {attempt+1}] litellm failed: {e}")
            
            # Fallback to openai
            if content is None:
                try:
                    import openai
                    client = openai.AsyncOpenAI()
                    resp = await client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                    content = resp.choices[0].message.content
                except Exception as e:
                    print(f"[Attempt {attempt+1}] openai failed: {e}")
            
            if content is None:
                continue
            
            # Parse response
            try:
                usage = extract_usage(resp)
                result = extract_json(content)
                key_points = result.pop("key_points")
                
                if not isinstance(result, dict):
                    raise ValueError(f"Expected dict, got {type(result)}")
                
                if len(result) != len(generations):
                    raise ValueError(f"[Attempt {attempt+1}] Expected {len(generations)} scores, got {len(result)}")
                
                # Extract scores
                scores = []
                for i, value in result.items():
                    if not isinstance(value, dict) or ("score" not in value) or ("thought" not in value):
                        raise ValueError(f"Item {i} missing 'score' field")
                    score = float(value["score"])
                    scores.append(min(max(score, 0.0), 1.0))
                break
                
            except Exception as e:
                print(f"[Attempt {attempt+1}] Failed to parse response: {e} | {content}")
                USER = os.getenv("USER", "unknown_user")
                with open(f"//llm_twin/log_state_reward_{USER}.out", "a") as f:
                    f.write(f"[Attempt {attempt+1}] Parse error: {e}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Generations: {generations_text}\n")
                    f.write("-" * 80 + "\n")
                
                if attempt < max_retry - 1:
                    await asyncio.sleep(1)
                else:
                    raise ValueError(f"All {max_retry} attempts failed to get valid scores")
        
        scores_list.append(scores)
    
    if len(scores_list) == 1:
        final_score = scores_list[0]
    final_score = [sum(score[i] for score in scores_list) / len(scores_list) for i in range(len(generations))]
    
    return [
        {
            "score": s, 
            "metrics_info": json.dumps({"key_points": key_points, "thought": result[i+1]})
        } for i, s in enumerate(final_score)
    ]

