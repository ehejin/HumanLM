import asyncio
import json
import os
import re
import time
from typing import Optional

from recipe.usim.utils import extract_json, parse_messages

HIERARCHY_PROMPT_BATCHED = '''You are a helpful and meticulous evaluator. \
Your task is to score how well the generated response matches the ground truth response with respect to each aspect below.

Aspects (aspect_i: description_i):
{hierarchy_desc}

You will be given the context, the ground truth response, and ONE generated response.

Provided Information:
<|The Start of Context|>
{context}
<|The End of Context|>

<|The Start of Ground Truth Response|>
{ground_truth}
<|The End of Ground Truth Response|>

{generation_text}

CRITICAL PRE-CHECK - Speaker Perspective Check (MUST DO FIRST):
Before scoring each aspect, verify that the generated response captures the HUMAN's perspective, not the assistant's. This check applies to aspects that describe user intent, goals, or stance.

The task is to capture what the HUMAN (user) expressed. The generation should describe the HUMAN's perspective, NOT the assistant's response or behavior.

Signs the generation incorrectly captures the ASSISTANT's perspective (automatic score = 0 for relevant aspects):
- Describes helping, explaining, assisting, or responding to the user
- Uses framing like "provide information about...", "help the user with...", "address the user's concern about..."
- Captures a VERY neutral, balanced, or hedged perspective typical of assistant responses when the ground truth shows a clear human stance
- Describes the assistant's intent or behavior rather than the HUMAN's

Examples of assistant-perspective errors:
- HUMAN asks "Can you explain how photosynthesis works?" Wrong generation: "Photosynthesis is the process by which plants convert sunlight into energy..." (assistant's explanation). Correct: the gt question itself "Can you explain how photosynthesis works?"
- HUMAN says "squeeze this text to 200 words: [long text]" Wrong generation: a 200-word summary of the text (assistant completing the task). Correct: the gt request itself "squeeze this text to 200 words: [long text]"

Note: Sometimes the HUMAN and assistant may genuinely share similar perspectives - this is fine. The check is specifically to catch when the generation describes the assistant's perspective INSTEAD OF the HUMAN's.

Apply this check to each aspect independently. If a particular aspect fails this check, assign score = 0 for all aspects and note the failure in the thought field.

Scoring Criteria (for each aspect_i, only if pre-check passes):
Assign a score in [0, 1] based on how much the aspects of generated response and the ground truth matches.

For each aspect_i, follow this procedure:
1. Extract 1-3 key points:
   - Extract K key points from the ground truth response only about the aspect based on the description (e.g., if evaluating a "stance", pick key points related to the stance like "clearly disagrees with X").
   - Each key point should be specific and distinct.

2. Score how well the generated response matches each key point:
   - For each key point i, compare it with the generated response and assign a match value m_i in range [0, 1]:
     - 1.0: The key point is precisely and perfectly reflected.
     - [0.7, 0.9]: Mostly reflected with small imperfections.
     - [0.4, 0.6]: Partially reflected or vague, but still leaning in the correct direction.
     - [0.1, 0.3]: Very weak reflection.
     - 0.0: Missed, contradicted, or reversed.

3. Compute coverage C = (m_1 + m_2 + ... + m_K) / K, which measures how comprehensive the generated response reflects the ground truth response.

4. Compute penalty P for extra or conflicting content:
   - Examine additional content in the generated response beyond those key points:
     - Does it introduce unsupported evidence and assumptions?
     - Is it irrelevant to what ground truth response expresses?
   - Set a penalty P ∈ [0, 1]:
     - 0.0: No problematic extra content; everything is perfectly matched.
     - [0.1, 0.3]: Slightly unnecessary or mildly speculative detail; meaning essentially unchanged.
     - [0.4, 0.6]: Moderate speculative or irrelevant content that somewhat shifts emphasis or adds unsupported ideas.
     - [0.7, 0.9]: Significant speculative, misleading, or conflicting content that clearly changes the meaning.
     - 1.0: Mostly off-topic, contradictory, or dominated by incorrect/hallucinated content.

5. If you are evaluating generated responses (skip if aspect_i is not response):
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
    "<aspect_1>": {{"thought": "<include (1) analysis of key points from ground truth along the aspect_1 dimension; (2) how well the generated response matches each key point; (3) compute the final score>", "score": <score>}},
    "<aspect_2>": {{"thought": "<include (1) analysis of key points from ground truth along the aspect_2 dimension; (2) how well the generated response matches each key point; (3) compute the final score>", "score": <score>}},
    "<aspect_3>": ...
}}

Format Notes:
- Make sure to score the responses with respect to each aspect and consider only one aspect at a time.
- All text in "thought" fields MUST be on a single line with no line breaks or newlines
- Use standard JSON string format with double quotes. For any quotes needed inside strings, use single quotes (')
- Double check the JSON array's format, especially for the comma and quotation marks
- Ensure that ALL fields, especially "thought" and "score", are present for each item
- You must provide exactly 1 score for each of the aspects: {hierarchy_names}.

Your evaluation:
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



# MAYBE REMOVE
# # amazon dataset store names have spaces
# def is_valid_gpt_name(name: str) -> bool:
#     return bool(name) and not INVALID_NAME_CHARS_RE.search(name)

# TODO: CHANGE TO ANY API MODELS
async def get_api_model_generation(raw_prompt: list, max_retry=2) -> Optional[str]:
    """
    get gen for gpt-5-mini. We try medium reasoning effort max_retry times
    then try low once then minimal once

    raw_prompt is list of messages from create_any_dataset in the format
    [{"role": ..., "content": ..., "name": ... (optional)}, ...]
    max_retry: Number of retries per reasoning effort level
    """
    import asyncio

    import openai

    client = openai.AsyncOpenAI()

    input_messages = []
    for msg in raw_prompt:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        m = {"role": role, "content": content}
        name = msg.get("name", None)
        if name is not None:
            if not is_valid_gpt_name(name):
                print(name)
                name = name.replace(" ", "_")
        m["name"] = name

        input_messages.append(m)

    reasoning_efforts = ["medium", "low", "minimal"]

    for effort in reasoning_efforts:
        if effort == "medium":
            max_retry = max_retry
        else:
            max_retry = 1

        for attempt in range(max_retry):
            try:
                completion = await client.chat.completions.create(
                    model="gpt-5-mini", # BUG: make this configurable?  
                    messages=input_messages,
                    reasoning_effort=effort,
                )

                # Extract text from the response
                output_text = (completion.choices[0].message.content or "")

                if output_text.strip():
                    return output_text.strip()
                else:
                    print(
                        f"[GPT-5-mini] Attempt {attempt+1}/{max_retry} with effort={effort}: Empty generation"
                    )

            except Exception as e:
                print(
                    f"[GPT-5-mini] Attempt {attempt+1}/{max_retry} with effort={effort} failed: {e}"
                )
                if attempt < max_retry - 1:
                    await asyncio.sleep(1)
        print(
            f"[GPT-5-mini] All {max_retry} attempts with effort={effort} failed, trying next effort level..."
        )

    print("[GPT-5-mini] All reasoning effort levels exhausted, returning None")
    return None


async def compute_score(data_source, generation, ground_truth, extra_info, config_path, **kwargs) -> list[float]:
    """
    Batched version: scores multiple generations at once.
    Returns a list of floats, one score per generation.
    """

    # in this case generations is a list of 1 generation
    run_api_model = kwargs.pop("run_api_model", False)
    if run_api_model:
        raw_prompt = json.loads(extra_info["raw_prompt"])
        gpt_generation = await get_api_model_generation(raw_prompt, max_retry=3)
        
        if gpt_generation is None:
            raise ValueError("OUTPUT FROM GPT-5-MINI FAILED")
        
        generations = [gpt_generation]
    
    max_retry = kwargs.pop("max_retry", 5)
    num_repeats = kwargs.pop("num_repeats", 1)
    # read json from config_path

    import json
    hierarchy_config = json.load(open(config_path, "r"))
    hierarchy_names = [h for h in hierarchy_config.keys()]
    hierarchy_descs = [v['desc'] for v in hierarchy_config.values()]
    hierarchy_dict = {h: d for h, d in zip(hierarchy_names, hierarchy_descs)}

    raw_prompt = json.loads(extra_info["raw_prompt"])
    context = parse_messages(raw_prompt)
    
    # Format all generations as a dict
    generation_text = f"<|The Start of Generated Response|>\n{generation}\n<|The End of Generated Response|>"
    
    other_guidelines = ""
    if 'other_guidelines' in kwargs:
        other_guidelines = kwargs.pop('other_guidelines')

    prompt = HIERARCHY_PROMPT_BATCHED.format(
        context=context,
        ground_truth=ground_truth,
        generation_text=generation_text,
        other_guidelines=other_guidelines,
        hierarchy_names=hierarchy_names,
        hierarchy_desc=json.dumps(hierarchy_dict, indent=2)
    )
    
    model = kwargs.pop("model", "openai/gpt-5-mini")
    # slight randomness to avoid getting stuck in a loop
    temperature = kwargs.pop("temperature", 0.1)
    max_tokens = kwargs.pop("max_tokens", 4096)

    print(f'state_reward_on_response  model {model} temperature {temperature} max_tokens {max_tokens}')
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
                print(result)
                
                if not isinstance(result, dict):
                    raise ValueError(f"Expected dict, got {type(result)}")
                
                # Extract scores
                score_dict = {}
                for i, value in result.items():
                    if not isinstance(value, dict) or "score" not in value:
                        raise ValueError(f"Item {i} missing 'score' field")
                    score = float(value["score"])
                    score = min(max(score, 0.0), 1.0)
                    score_dict[i] = score

                if len(score_dict) != len(hierarchy_names):
                    raise ValueError(f"[Attempt {attempt+1}] Expected {len(hierarchy_names)} scores, got {len(score_dict)}")
                assert set(score_dict.keys()) == set(hierarchy_names), f"Score keys {score_dict.keys()} do not match expected {hierarchy_names}"
                break
                
            except Exception as e:
                print(f"[Attempt {attempt+1}] Failed to parse response: {e}")
                import os
                USER = os.getenv("USER", "unknown_user")
                with open(f"//llm_twin/log_state_reward_on_response_{USER}.out", "a") as f:
                    f.write(f"[Attempt {attempt+1}] Parse error: {e}\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Generations: {generation_text}\n")
                    f.write("-" * 80 + "\n")

                if attempt < max_retry - 1:
                    await asyncio.sleep(1)
                else:
                    raise ValueError(f"All {max_retry} attempts failed to get valid scores")
        
    score_dict.update({
        "metrics_info": content
    })
    return score_dict