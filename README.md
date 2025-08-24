# multimodal-search

# CLIP Image Search Engine with Re-ranking

A semantic image search system using **CLIP (ViT-B/32)** and **Flickr30k dataset** with neural re-ranking for improved results.

## What it does

Searches through **31,014 Flickr30k images** using natural language queries:

1. **CLIP Encoding** - Extracts image and text embeddings using pre-trained CLIP model
2. **FAISS Search** - Fast similarity search with cosine similarity  
3. **Neural Re-ranking** - Trains custom neural network to improve search relevance

## Files

- `data_processor.py` - Extracts and saves CLIP embeddings from Flickr30k
- `search_engine.py` - Main search engine with re-ranking functionality
- `data/` - Stored embeddings and metadata
- `test/` - Sample search results (original vs re-ranked)

## Quick Start

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Process dataset:
```bash
python data_processor.py    # Extract embeddings (takes time)
python search_engine.py     # Run search engine
```

## Sample Results

**Query: "a dog running in the park"**

| Method | Top Result | CLIP Score | ReRank Score |
|--------|------------|------------|--------------|
| Original | "The brown dog is walking along a grassy path with its tongue out." | 0.361 | 0.992 |
| Re-ranked | "A large furry brown dog is walking with a leash in his mouth." | 0.335 | 1.000 |

The re-ranker identifies more relevant results by learning semantic relationships beyond CLIP similarity.

## Performance

- **Dataset**: 31,014 Flickr30k test images
- **Embedding Size**: 512D CLIP features  
- **Search Speed**: Near real-time with FAISS indexing
- **Re-ranker Training**: 1800 pairs (450 positive, 1350 negative)
- **Final Loss**: 0.0010 after 40 epochs
