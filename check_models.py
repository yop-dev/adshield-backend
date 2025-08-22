import json

with open('deepfake_models.json') as f:
    data = json.load(f)
    
print("Available deepfake detection models on Hugging Face:")
print("-" * 60)
for model in data[:15]:
    model_id = model.get('id', '')
    downloads = model.get('downloads', 0)
    likes = model.get('likes', 0)
    print(f"{model_id}")
    print(f"  Downloads: {downloads:,}, Likes: {likes}")
    print()
