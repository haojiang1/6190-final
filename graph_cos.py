import torch
from transformers import CLIPTokenizer, CLIPTextModel
import numpy as np
import clip
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix: D^(-1/2) A D^(-1/2)
    """
    # Calculate degree matrix D
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # Symmetric normalization
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return normalized_adj


def preprocess_adj(adj_matrix, num_nodes):
    """
    Preprocessing of adjacency matrix for GCN and conversion to tuple representation.
    """
    # Add self-loops
    # adj = adj_matrix + np.eye(num_nodes)

    # Convert to sparse matrix
    adj_sparse = sp.csr_matrix(adj_matrix)

    # Normalize adjacency matrix
    adj_normalized = normalize_adj(adj_sparse)
    adj_normalized_dense = adj_normalized.toarray()
    # print(adj_normalized_dense)
    # Convert to torch sparse tensor
    # adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)

    return adj_normalized_dense

def visualize_adjacency_matrix(matrix, categories):
    plt.figure(figsize=(15, 12))

    # Create DataFrame for better visualization
    df = pd.DataFrame(matrix, index=categories, columns=categories)

    # Create heatmap
    sns.heatmap(df, annot=True, cmap='YlOrRd', fmt='.1f',
                square=True, cbar_kws={'label': 'Relationship Strength'})

    plt.title('VOC Categories Adjacency Matrix', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
def get_category_adjacency_matrix(categories):
    # Initialize CLIP text model and tokenizer
    # model_name = "openai/clip-vit-base-patch32"
    # tokenizer = CLIPTokenizer.from_pretrained(model_name)
    # text_model = CLIPTextModel.from_pretrained(model_name)

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # text_model = text_model.to(device)

    # Create empty adjacency matrix
    adj_matrix = np.zeros((len(categories), len(categories)))

    # Prepare text inputs
    cat_lists = [cat for cat in categories]
    print(cat_lists)
    text_inputs = cat_lists

    # Tokenize texts
    model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(cat_lists).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    print("sim", text_features[0]*text_features[1])

    # Calculate cosine similarity matrix
    similarity_matrix = torch.mm(text_features, text_features.t())
    # similarity_matrix = F.softmax(similarity_matrix, dim=1)
    # similarity_matrix = F.softmax(similarity_matrix, dim=1)

    # Convert to numpy array
    adj_matrix = similarity_matrix.cpu().numpy()

    return adj_matrix, categories


# Example usage
if __name__ == "__main__":
    categories = [
        'CLS', 'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
        'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
        'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
    ]

    # Get adjacency matrix
    adj_matrix, categories = get_category_adjacency_matrix(categories)

    # Print matrix shape
    print(f"Adjacency matrix shape: {adj_matrix.shape}")

    # Create a dictionary mapping categories to their index
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

    # Example: Print similarity between some categories
    example_pairs = [
        ('dog', 'cat'),
        ('car', 'bus'),
        ('person', 'chair'),
        ('bird', 'aeroplane'),
        ('bus', 'dog')
    ]

    print("\nSimilarity scores for example pairs:")
    for cat1, cat2 in example_pairs:
        idx1, idx2 = cat_to_idx[cat1], cat_to_idx[cat2]
        print(f"{cat1} - {cat2}: {adj_matrix[idx1][idx2]:.3f}")

    # Optional: Print full matrix with category labels
    print("\nFull adjacency matrix:")
    print("Categories:", categories)
    print(adj_matrix)
    weighted_matrix = preprocess_adj(adj_matrix, len(adj_matrix))
    print(weighted_matrix)
    visualize_adjacency_matrix(weighted_matrix, categories)
    np.save('VOC2017_cos.npy', weighted_matrix)
