import os
import numpy as np
import torch
import pickle
import faiss
import clip
from PIL import Image
import shutil
from datasets import load_dataset

class ImageSearchEngine:
    def __init__(self, data_dir="data", model_name="ViT-B/32"):
        """Initialize search engine"""
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Loading CLIP model...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        print("Loading Flickr30k dataset for image access...")
        self.dataset = load_dataset("nlphuji/flickr30k", split="test")
        self.load_data()
        self.build_index()
    
    def load_data(self):
        """Load precomputed embeddings and metadata"""
        print("Loading embeddings and metadata...")
        
        self.image_embeddings = np.load(f"{self.data_dir}/image_embeddings.npy")
        self.text_embeddings = np.load(f"{self.data_dir}/text_embeddings.npy")
        
        with open(f"{self.data_dir}/metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
            self.image_paths = metadata['image_paths']
            self.captions = metadata['captions']
            self.embedding_dim = metadata['embedding_dim']
        
        print(f"Loaded {len(self.image_paths)} images with {self.embedding_dim}D embeddings")
    
    def build_index(self):
        """Build FAISS index for fast similarity search"""
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.image_embeddings.astype('float32'))
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def encode_query(self, query_text):
        with torch.no_grad():
            text_tokens = clip.tokenize([query_text], truncate=True).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        return text_features.cpu().numpy()
    
    def search(self, query_text, top_k=10):
        query_embedding = self.encode_query(query_text)
        similarities, indices = self.index.search(query_embedding.astype('float32'), top_k)
    
        results = []
        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            results.append({
                'rank': i + 1,
                'image_path': self.image_paths[idx],
                'caption': self.captions[idx],
                'similarity': float(sim),
                'image_idx': int(idx)
            })
        
        return results
    
    def get_image_path(self, image_path):
        return os.path.join(self.data_dir, "images", image_path)


class ReRanker:
    def __init__(self, embedding_dim=512, hidden_dim=256):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1),
            torch.nn.Sigmoid()
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.BCELoss()
    
    def create_training_data(self, search_engine, num_queries=100):
        print("Creating training data for re-ranker...")
        
        query_embeddings = []
        image_embeddings = []
        labels = []
        np.random.seed(42)
        query_indices = np.random.choice(len(search_engine.captions), num_queries, replace=False)
        
        for query_idx in query_indices:
            query_text = search_engine.captions[query_idx]
            query_emb = search_engine.encode_query(query_text)
            
            pos_img_emb = search_engine.image_embeddings[query_idx:query_idx+1]
            query_embeddings.append(query_emb)
            image_embeddings.append(pos_img_emb)
            labels.append(1.0)
            
            neg_indices = np.random.choice(len(search_engine.image_embeddings), 3, replace=False)
            for neg_idx in neg_indices:
                if neg_idx != query_idx:
                    neg_img_emb = search_engine.image_embeddings[neg_idx:neg_idx+1]
                    query_embeddings.append(query_emb)
                    image_embeddings.append(neg_img_emb)
                    labels.append(0.0)
        
        self.train_query_embs = np.vstack(query_embeddings)
        self.train_image_embs = np.vstack(image_embeddings)
        self.train_labels = np.array(labels)
        
        print(f"Created {len(self.train_labels)} training pairs "
              f"({np.sum(self.train_labels)} positive, {len(self.train_labels) - np.sum(self.train_labels)} negative)")
    
    def train(self, epochs=50, batch_size=32):
        print("Training re-ranker...")
        
        dataset_size = len(self.train_labels)
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(dataset_size)
            total_loss = 0
            num_batches = 0
            
            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i:i+batch_size]
                
                query_batch = torch.FloatTensor(self.train_query_embs[batch_indices]).to(self.device)
                image_batch = torch.FloatTensor(self.train_image_embs[batch_indices]).to(self.device)
                label_batch = torch.FloatTensor(self.train_labels[batch_indices]).to(self.device)
                
                # Concatenate query and image embeddings
                combined = torch.cat([query_batch, image_batch], dim=1)
                
                # Forward pass
                predictions = self.model(combined).squeeze()
                loss = self.criterion(predictions, label_batch)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def rerank(self, query_embedding, image_embeddings):
        self.model.eval()
        
        with torch.no_grad():
            query_batch = np.repeat(query_embedding, len(image_embeddings), axis=0)
            
            query_tensor = torch.FloatTensor(query_batch).to(self.device)
            image_tensor = torch.FloatTensor(image_embeddings).to(self.device)
            
            combined = torch.cat([query_tensor, image_tensor], dim=1)
            scores = self.model(combined).squeeze()
            
        return scores.cpu().numpy()
    
    def save(self, path="reranker_model.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Re-ranker saved to {path}")
    
    def load(self, path="reranker_model.pth"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Re-ranker loaded from {path}")


def main():
    engine = ImageSearchEngine()

    queries = [
        "a dog running in the park",
        "people sitting on a beach",
        "a red car on the street",
        "children playing football"
    ]
    
    print("\n" + "="*50)
    print("TESTING BASIC SEARCH")
    print("="*50)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = engine.search(query, top_k=5)
        
        for r in results:
            print(f"  {r['rank']}. File: {r['image_path']} - Score: {r['similarity']:.3f} - {r['caption']}")
    
    print("\n" + "="*50)
    print("TRAINING RE-RANKER")
    print("="*50)
    
    reranker = ReRanker(embedding_dim=engine.embedding_dim)
    reranker.create_training_data(engine, num_queries=200)
    reranker.train(epochs=40, batch_size=64)
    
    print("\n" + "="*50)
    print("TESTING WITH RE-RANKING")
    print("="*50)
    
    test_query = "a dog running in the park"
    results = engine.search(test_query, top_k=20)
    
    query_emb = engine.encode_query(test_query)
    result_image_embs = np.array([engine.image_embeddings[r['image_idx']] for r in results])

    rerank_scores = reranker.rerank(query_emb, result_image_embs)
    
    for i, score in enumerate(rerank_scores):
        results[i]['rerank_score'] = float(score)
    
    results_reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
    
    print(f"\nOriginal top 5 results for: '{test_query}'")
    for i, r in enumerate(results[:5]):
        print(f"  {i+1}. CLIP: {r['similarity']:.3f}, File: {r['image_path']} - ReRank: {r['rerank_score']:.3f} - {r['caption']}")
    
    print(f"\nRe-ranked top 5 results:")
    for i, r in enumerate(results_reranked[:5]):
        print(f"  {i+1}. CLIP: {r['similarity']:.3f}, File: {r['image_path']} - ReRank: {r['rerank_score']:.3f} - {r['caption']}")

    reranker.save(path="reranker.pth")

    def save_top_images(results, folder, dataset):
        os.makedirs(folder, exist_ok=True)
        for i, r in enumerate(results[:5]):
            # image_idx corresponds to index in dataset
            img = dataset[r['image_idx']]['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            filename = f"rank{i+1}_{r['image_path']}"
            save_path = os.path.join(folder, filename)
            img.save(save_path)
    
    save_top_images(results[:5], "test/original", engine.dataset)

    save_top_images(results_reranked[:5], "test/rerank", engine.dataset)

    print("\nSaved top 5 images for original ranking in 'test/original/'")
    print("Saved top 5 images for reranked results in 'test/rerank/'")


if __name__ == "__main__":
    main()