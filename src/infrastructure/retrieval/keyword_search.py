"""BM25-based keyword search for hybrid retrieval."""

from typing import List, Tuple
from rank_bm25 import BM25Okapi


def bm25_search(
    queries: List[str],
    documents: List[str],
    top_k: int
) -> List[Tuple[int, float]]:
    """
    Perform BM25 keyword search on documents.
    
    Args:
        queries: List of query strings (will be combined for keyword extraction)
        documents: List of document texts to search in
        top_k: Number of top documents to return
        
    Returns:
        List of tuples (document_index, bm25_score) sorted by score descending
    """
    if not documents:
        return []
    
    if not queries:
        return []
    
    # Tokenize documents (BM25Okapi does this automatically)
    tokenized_docs = [doc.split() for doc in documents]
    
    # Create BM25 index
    bm25 = BM25Okapi(tokenized_docs)
    
    # Combine all queries into one search query
    # Extract keywords from all queries
    combined_query = " ".join(queries)
    tokenized_query = combined_query.split()
    
    # Get BM25 scores for all documents
    scores = bm25.get_scores(tokenized_query)
    
    # Create list of (index, score) tuples
    indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]
    
    # Sort by score descending
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_k results
    return indexed_scores[:top_k]


def bm25_search_all(
    queries: List[str],
    documents: List[str]
) -> List[float]:
    """
    Perform BM25 keyword search on all documents and return scores for all.
    
    Args:
        queries: List of query strings (will be combined for keyword extraction)
        documents: List of document texts to search in
        
    Returns:
        List of BM25 scores for all documents (one score per document, in same order)
    """
    if not documents:
        return []
    
    if not queries:
        return [0.0] * len(documents)
    
    # Tokenize documents (BM25Okapi does this automatically)
    tokenized_docs = [doc.split() for doc in documents]
    
    # Create BM25 index
    bm25 = BM25Okapi(tokenized_docs)
    
    # Combine all queries into one search query
    # Extract keywords from all queries
    combined_query = " ".join(queries)
    tokenized_query = combined_query.split()
    
    # Get BM25 scores for all documents
    scores = bm25.get_scores(tokenized_query)
    
    # Return all scores as list of floats
    return [float(score) for score in scores]


def normalize_bm25_scores(scores: List[float]) -> List[float]:
    """
    Normalize BM25 scores to [0, 1] range using Min-Max normalization.
    
    Args:
        scores: List of BM25 scores
        
    Returns:
        List of normalized scores in [0, 1] range
    """
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    # Handle case where all scores are the same
    if max_score == min_score:
        return [1.0] * len(scores)
    
    # Min-Max normalization: (score - min) / (max - min)
    normalized = [(score - min_score) / (max_score - min_score) for score in scores]
    
    return normalized

