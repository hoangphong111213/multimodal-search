import os
import numpy as np
import torch
import pickle
import clip
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm


class DataProcessor:
    def __init__(self, model_name="ViT-B/32"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        
        print(f"Loaded CLIP {model_name} model")
    
    def load_flickr30k(self, split="test"):
        print(f"Loading Flickr30k {split} split...")
        dataset = load_dataset("nlphuji/flickr30k", split=split)
        
        print(f"Loaded full samples")
        return dataset
    
    def extract_image_embeddings(self, dataset, batch_size=32):
        print("Extracting image embeddings...")
        
        image_embeddings = []
        image_paths = []
        
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch_end = min(i + batch_size, len(dataset))
            batch_images = []
            batch_paths = []
            
            for j in range(i, batch_end):
                image = dataset[j]['image']
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                processed_image = self.preprocess(image)
                batch_images.append(processed_image)
                batch_paths.append(f"image_{j}.jpg")
            
            image_tensor = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize
                
            image_embeddings.append(image_features.cpu().numpy())
            image_paths.extend(batch_paths)
        
        return np.vstack(image_embeddings), image_paths
    
    def extract_text_embeddings(self, dataset, batch_size=64):
        print("Extracting text embeddings...")
        
        text_embeddings = []
        captions = []
        
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch_end = min(i + batch_size, len(dataset))
            batch_captions = []
            
            for j in range(i, batch_end):
                caption = dataset[j]['caption'][0] if dataset[j]['caption'] else ""
                batch_captions.append(caption)
                captions.append(caption)
            
            text_tokens = clip.tokenize(batch_captions, truncate=True).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize
            
            text_embeddings.append(text_features.cpu().numpy())
        
        return np.vstack(text_embeddings), captions
    
    def save_embeddings(self, image_embeddings, text_embeddings, image_paths, captions, 
                       dataset, save_dir="data"):
        os.makedirs(save_dir, exist_ok=True)
        
        np.save(f"{save_dir}/image_embeddings.npy", image_embeddings)
        np.save(f"{save_dir}/text_embeddings.npy", text_embeddings)
        
        metadata = {
            'image_paths': image_paths,
            'captions': captions,
            'embedding_dim': image_embeddings.shape[1]
        }
        
        with open(f"{save_dir}/metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        images_dir = f"{save_dir}/images"
        os.makedirs(images_dir, exist_ok=True)
        
        print("Saving sample images...")
        for i, image_path in enumerate(tqdm(image_paths[:100])):  # Save first 100 images
            image = dataset[i]['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(f"{images_dir}/{image_path}")
        
        print(f"Data saved to {save_dir}/")


def main():
    """Main processing pipeline"""
    processor = DataProcessor()
    dataset = processor.load_flickr30k(split="test")

    image_embeddings, image_paths = processor.extract_image_embeddings(dataset)
    text_embeddings, captions = processor.extract_text_embeddings(dataset)
    
    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    
    processor.save_embeddings(image_embeddings, text_embeddings, 
                            image_paths, captions, dataset)
    
    print("Processing complete!")


if __name__ == "__main__":
    main()