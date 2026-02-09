import asyncio
import torch
import openai
import re

# Global client instance to avoid connection issues
_openai_client = None

def get_openai_client():
    """Get or create the global OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.AsyncOpenAI()
    return _openai_client


async def get_openai_embedding_async(
    text: str,
    model: str = "text-embedding-3-large",
    max_retry: int = 3,
) -> torch.FloatTensor:
    """
    Get OpenAI embedding asynchronously for a single text.
    
    Args:
        text: Input text to embed
        model: OpenAI embedding model to use
        max_retry: Maximum number of retry attempts
        
    Returns:
        torch.FloatTensor: Embedding vector (embedding_dim,)
    """
    assert isinstance(text, str) and len(text) > 0, "text must be non-empty string"
    
    client = get_openai_client()
    
    for attempt in range(max_retry):
        try:
            response = await client.embeddings.create(
                input=[text],
                model=model
            )
            embedding = torch.FloatTensor(response.data[0].embedding)
            return embedding
        except openai.BadRequestError as e:
            print(f'{e}')
            e_str = str(e)
            ori_length = len(text.split(' '))
            match = re.search(r'maximum context length is (\d+) tokens, however you requested (\d+) tokens', e_str)
            if match is not None:
                max_length = int(match.group(1))
                cur_length = int(match.group(2))
                ratio = float(max_length) / cur_length
                for reduce_rate in range(9, 0, -1):
                    shorten_text = text.split(' ')
                    length = int(ratio * ori_length * (reduce_rate * 0.1))
                    shorten_text = ' '.join(shorten_text[:length])
                    try:
                        response = await client.embeddings.create(input=[shorten_text], model=model)
                        print(f'length={length} works! reduce_rate={0.1 * reduce_rate}.')
                        embedding = torch.FloatTensor(response.data[0].embedding)
                        return embedding
                    except:
                        continue
            raise
        except openai.RateLimitError as e:
            if attempt < max_retry - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limit hit, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise
        except Exception as e:
            if attempt < max_retry - 1:
                print(f"Error: {e}, retrying...")
                await asyncio.sleep(1)
            else:
                raise
    
    raise RuntimeError(f"Failed to get embedding after {max_retry} retries")


async def compute_score(data_source, generation, ground_truth, extra_info=None, **kwargs):
    """
    Compute cosine similarity score between generation and ground truth using embeddings.
    
    Args:
        data_source: Data source identifier (not used, for compatibility)
        generation: Generated text to evaluate (str)
        ground_truth: Reference text (str)
        extra_info: Additional information (optional)
        **kwargs: Additional arguments including:
            - model: OpenAI embedding model to use (default: "text-embedding-3-large")
            - device: Device for GPU computation (default: "cpu")
        
    Returns:
        float: Cosine similarity score
    """
    model = kwargs.get("model", "text-embedding-3-large")
    device = kwargs.get("device", "cpu")
    
    # Get embeddings asynchronously
    gen_task = get_openai_embedding_async(generation, model)
    gt_task = get_openai_embedding_async(ground_truth, model)
    gen_embedding, gt_embedding = await asyncio.gather(gen_task, gt_task)
    
    # Move to device and compute cosine similarity
    # Embeddings are already normalized, so dot product = cosine similarity
    gen_emb = gen_embedding.to(device)
    gt_emb = gt_embedding.to(device)
    
    similarity = torch.dot(gen_emb, gt_emb)
    
    return similarity.item()


# Example usage
if __name__ == "__main__":
    async def main():
        score = await compute_score(
            data_source="test",
            generation="The cat sat on the mat."*10000,
            ground_truth="A feline rested on the rug."*10000
        )
        print(f"Similarity score: {score:.4f}")
        
        # With custom device
        score_gpu = await compute_score(
            data_source="test",
            generation="The cat sat on the mat."*10000,
            ground_truth="A feline rested on the rug."*10000,
            device="cuda"
        )
        print(f"Similarity score (GPU): {score_gpu:.4f}")
    
    asyncio.run(main())