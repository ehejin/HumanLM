import torch
from sentence_transformers import SentenceTransformer
from threading import Lock
import asyncio
from functools import partial

_SNOWFLAKE_MODELS = {}
_SNOWFLAKE_LOCK = Lock()
_SNOWFLAKE_INIT_LOCK = Lock()


def _get_snowflake_model(model_name: str = "Snowflake/snowflake-arctic-embed-l-v2.0", device: str = "cpu"):
    cache_key = f"{model_name}_{device}"
    
    with _SNOWFLAKE_LOCK:
        if cache_key not in _SNOWFLAKE_MODELS:
            print(f"Loading Snowflake embedding model: {model_name} on {device}...")
            
            with _SNOWFLAKE_INIT_LOCK:
                if cache_key not in _SNOWFLAKE_MODELS:
                    try:
                        _SNOWFLAKE_MODELS[cache_key] = SentenceTransformer(model_name, device=device)
                        print(f"Successfully loaded Snowflake model: {model_name}")
                    except Exception as e:
                        print(f"Error loading Snowflake model: {e}")
                        raise
        
        return _SNOWFLAKE_MODELS[cache_key]


def compute_score_sync(data_source, generation, ground_truth, extra_info=None, **kwargs):
    model_name = kwargs.get("model_name", "Snowflake/snowflake-arctic-embed-l-v2.0")
    device = kwargs.get("device", "cpu")
    use_query_prefix = kwargs.get("use_query_prefix", False)
    
    model = _get_snowflake_model(model_name, device)
    
    gen_text = f"query: {generation}" if use_query_prefix else generation
    texts = [gen_text, ground_truth]
    
    embeddings = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    gen_embedding = embeddings[0]
    gt_embedding = embeddings[1]
    
    similarity = torch.dot(gen_embedding, gt_embedding)
    return similarity.item()


async def compute_score(data_source, generation, ground_truth, extra_info=None, **kwargs) -> float:
    '''
    Async wrapper
    '''
    loop = asyncio.get_running_loop()
    func = partial(compute_score_sync, data_source, generation, ground_truth, extra_info, **kwargs)
    return await loop.run_in_executor(None, func)



if __name__ == "__main__":
    score = compute_score(
        data_source="test",
        generation="The cat sat on the mat.",
        ground_truth="A feline rested on the rug."
    )
    print(f"Similarity score: {score:.4f}")
    
    if torch.cuda.is_available():
        score_gpu = compute_score(
            data_source="test",
            generation="The cat sat on the mat.",
            ground_truth="A feline rested on the rug.",
            device="cuda"
        )
        print(f"Similarity score (GPU): {score_gpu:.4f}")
    
    score_query = compute_score(
        data_source="test",
        generation="what is snowflake?",
        ground_truth="Snowflake is a cloud data platform.",
        use_query_prefix=True
    )
    print(f"Similarity score (with query prefix): {score_query:.4f}")

