import os
import numpy as np
import torch
import pickle
import faiss
import clip
from PIL import Image
import shutil
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class ImageSearchEngine:
    def __init__(self, data_dir="data", model_name="ViT-B/32"):
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
    def __init__(self, embedding_dim=512, hidden_dim=128):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Simple MLP
        self.model = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2, hidden_dim),  # Concat query + image embeddings
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim // 4, 1),  # Output similarity score
            torch.nn.Sigmoid()
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.BCELoss()
    
    def create_training_data(self, search_engine, num_queries=100, val_split=0.2, test_split=0.2):
        print("Creating training, validation and test data for re-ranker...")
        np.random.seed(42)
        query_indices = np.random.choice(len(search_engine.captions), num_queries, replace=False)
        
        num_val = int(num_queries * val_split)
        num_test = int(num_queries * test_split)
        num_train = num_queries - num_val - num_test
        
        train_indices = query_indices[:num_train]
        val_indices = query_indices[num_train:num_train + num_val]
        test_indices = query_indices[num_train + num_val:]
        
        print(f"Data split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test queries")
        
        self.train_query_embs, self.train_image_embs, self.train_labels = self._create_pairs(search_engine, train_indices)
        self.val_query_embs, self.val_image_embs, self.val_labels = self._create_pairs(search_engine, val_indices)
        self.test_query_embs, self.test_image_embs, self.test_labels = self._create_pairs(search_engine, test_indices)
        
        print(f"Training: {len(self.train_labels)} pairs ({np.sum(self.train_labels)} positive)")
        print(f"Validation: {len(self.val_labels)} pairs ({np.sum(self.val_labels)} positive)")
        print(f"Test: {len(self.test_labels)} pairs ({np.sum(self.test_labels)} positive)")
    
    def _create_pairs(self, search_engine, query_indices):
        query_embeddings = []
        image_embeddings = []
        labels = []
        
        for query_idx in query_indices:
            query_text = search_engine.captions[query_idx]
            query_emb = search_engine.encode_query(query_text)
            
            # Positive example: caption matches its own image
            pos_img_emb = search_engine.image_embeddings[query_idx:query_idx+1]
            query_embeddings.append(query_emb)
            image_embeddings.append(pos_img_emb)
            labels.append(1.0)
            
            # Negative examples: caption with random images
            neg_indices = np.random.choice(len(search_engine.image_embeddings), 3, replace=False)
            for neg_idx in neg_indices:
                if neg_idx != query_idx:  # Avoid same image
                    neg_img_emb = search_engine.image_embeddings[neg_idx:neg_idx+1]
                    query_embeddings.append(query_emb)
                    image_embeddings.append(neg_img_emb)
                    labels.append(0.0)
        
        return np.vstack(query_embeddings), np.vstack(image_embeddings), np.array(labels)
    
    def evaluate(self, query_embs, image_embs, labels):
        self.model.eval()
        
        with torch.no_grad():
            query_tensor = torch.FloatTensor(query_embs).to(self.device)
            image_tensor = torch.FloatTensor(image_embs).to(self.device)
            label_tensor = torch.FloatTensor(labels).to(self.device)
            
            combined = torch.cat([query_tensor, image_tensor], dim=1)
            predictions = self.model(combined).squeeze()

            loss = self.criterion(predictions, label_tensor)
            
            binary_preds = (predictions > 0.5).float()
            
            accuracy = accuracy_score(labels, binary_preds.cpu().numpy())
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, binary_preds.cpu().numpy(), average='binary', zero_division=0
            )
            
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, epochs=50, batch_size=32):
        print("Training re-ranker with validation...")
        
        dataset_size = len(self.train_labels)
        best_val_f1 = 0.0
        best_model_state = None
        
        train_history = {'loss': [], 'accuracy': [], 'f1': []}
        val_history = {'loss': [], 'accuracy': [], 'f1': []}
        
        for epoch in range(epochs):
            self.model.train()
            
            indices = np.random.permutation(dataset_size)
            total_loss = 0
            num_batches = 0
            
            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i:i+batch_size]
                
                query_batch = torch.FloatTensor(self.train_query_embs[batch_indices]).to(self.device)
                image_batch = torch.FloatTensor(self.train_image_embs[batch_indices]).to(self.device)
                label_batch = torch.FloatTensor(self.train_labels[batch_indices]).to(self.device)
                
                combined = torch.cat([query_batch, image_batch], dim=1)
                
                predictions = self.model(combined).squeeze()
                loss = self.criterion(predictions, label_batch)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            train_metrics = self.evaluate(self.train_query_embs, self.train_image_embs, self.train_labels)
            val_metrics = self.evaluate(self.val_query_embs, self.val_image_embs, self.val_labels)
            
            train_history['loss'].append(train_metrics['loss'])
            train_history['accuracy'].append(train_metrics['accuracy'])
            train_history['f1'].append(train_metrics['f1'])
            
            val_history['loss'].append(val_metrics['loss'])
            val_history['accuracy'].append(val_metrics['accuracy'])
            val_history['f1'].append(val_metrics['f1'])
            
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_model_state = self.model.state_dict().copy()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nBest validation F1: {best_val_f1:.4f}")
        
        self.train_history = train_history
        self.val_history = val_history
        
        return train_history, val_history
    
    def test_evaluate(self):
        print("\nEvaluating on test set...")
        test_metrics = self.evaluate(self.test_query_embs, self.test_image_embs, self.test_labels)
        
        print("Test Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1-Score: {test_metrics['f1']:.4f}")
        
        return test_metrics
    
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
    print("TRAINING RE-RANKER WITH VALIDATION")
    print("="*50)
    
    reranker = ReRanker(embedding_dim=engine.embedding_dim)
    
    reranker.create_training_data(engine, num_queries=20000, val_split=0.05, test_split=0.05)
    
    train_history, val_history = reranker.train(epochs=30, batch_size=64)
    
    test_metrics = reranker.test_evaluate()
    
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
        print(f"  {i+1}. CLIP: {r['similarity']:.4f}, File: {r['image_path']} - ReRank: {r['rerank_score']:.4f} - {r['caption']}")
    
    print(f"\nRe-ranked top 5 results:")
    for i, r in enumerate(results_reranked[:5]):
        print(f"  {i+1}. CLIP: {r['similarity']:.4f}, File: {r['image_path']} - ReRank: {r['rerank_score']:.4f} - {r['caption']}")

    reranker.save(path="reranker.pth")

    def save_top_images(results, folder, dataset):
        os.makedirs(folder, exist_ok=True)
        for i, r in enumerate(results[:5]):
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
