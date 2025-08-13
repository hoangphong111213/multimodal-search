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
| Original | "brown dog walking along grassy path" | 0.361 | 0.276 |
| Re-ranked | "gray dog jumps in air to catch Frisbee" | 0.328 | 1.000 |

The re-ranker identifies more relevant results by learning semantic relationships beyond CLIP similarity.

## Performance

- **Dataset**: 31,014 Flickr30k test images
- **Embedding Size**: 512D CLIP features  
- **Search Speed**: Near real-time with FAISS indexing
- **Re-ranker Training**: 800 pairs (200 positive, 600 negative)
- **Final Loss**: 0.0010 after 40 epochs
