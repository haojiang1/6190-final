import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp

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
    adj = adj_matrix + np.eye(num_nodes)

    # Convert to sparse matrix
    adj_sparse = sp.csr_matrix(adj)

    # Normalize adjacency matrix
    adj_normalized = normalize_adj(adj_sparse)
    adj_normalized_dense = adj_normalized.toarray()
    # print(adj_normalized_dense)
    # Convert to torch sparse tensor
    # adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)

    return adj_normalized_dense
def create_voc_adjacency_matrix():
    # Define the 20 categories in order
    categories = [
        'CLS', 'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
        'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
        'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
    ]

    # Create a 20x20 matrix initialized with zeros
    matrix = np.zeros((21, 21))

    # Define relationships (1 indicates relationship exists)
    relationships = {
        # Living beings relationships
        ('person', 'bicycle'): 1,  # person can ride bicycle
        ('person', 'car'): 1,  # person can drive car
        ('person', 'chair'): 1,  # person uses chair
        ('person', 'sofa'): 1,  # person uses sofa
        ('person', 'diningtable'): 1,  # person uses table

        # Animal relationships
        ('cat', 'sofa'): 1,  # cat can be on sofa
        ('dog', 'sofa'): 1,  # dog can be on sofa
        ('bird', 'pottedplant'): 1,  # bird can be near plant

        # Vehicle relationships
        ('car', 'road'): 1,
        ('bus', 'road'): 1,
        ('motorbike', 'road'): 1,
        ('bicycle', 'road'): 1,

        # Indoor object relationships
        ('bottle', 'diningtable'): 1,
        ('chair', 'diningtable'): 1,
        ('tvmonitor', 'diningtable'): 1,
    }

    # Fill the matrix
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            # Check both directions of relationships
            if (cat1, cat2) in relationships:
                matrix[i][j] = relationships[(cat1, cat2)]
            if (cat2, cat1) in relationships:
                matrix[i][j] = relationships[(cat2, cat1)]

    return matrix, categories


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


def analyze_matrix(matrix, categories):
    # Calculate some basic metrics
    print("Matrix Analysis:")
    print(f"Total relationships: {np.sum(matrix)}")
    print(f"Network density: {np.sum(matrix) / (len(categories) * len(categories)):.3f}")

    # Find most connected categories
    connections = np.sum(matrix, axis=1)
    category_connections = [(cat, conn) for cat, conn in zip(categories, connections)]
    print("\nMost connected categories:")
    for cat, conn in sorted(category_connections, key=lambda x: x[1], reverse=True)[:5]:
        print(f"{cat}: {conn} connections")


def create_weighted_relationships():
    """Create weighted relationships based on co-occurrence likelihood"""
    # Define the 20 categories
    categories = [
        'CLS', 'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
        'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
        'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
    ]

    # Create matrix
    matrix = np.zeros((21, 21))

    # Define weighted relationships (values between 0 and 1)
    weighted_relationships = {
        # Strong relationships (0.8-1.0)
        ('person', 'chair'): 0.9,
        ('chair', 'diningtable'): 0.8,
        ('person', 'sofa'): 0.8,

        # Medium relationships (0.5-0.7)
        ('bottle', 'diningtable'): 0.7,
        ('tvmonitor', 'sofa'): 0.6,
        ('person', 'car'): 0.6,

        # Weak relationships (0.2-0.4)
        ('bird', 'pottedplant'): 0.3,
        ('cat', 'sofa'): 0.4,
        ('dog', 'sofa'): 0.4,

        # Additional contextual relationships
        ('car', 'motorbike'): 0.5,  # road vehicles
        ('bus', 'car'): 0.5,
        ('bicycle', 'person'): 0.6,
        ('horse', 'person'): 0.4,
    }

    # Fill the matrix with weighted relationships
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if (cat1, cat2) in weighted_relationships:
                matrix[i][j] = weighted_relationships[(cat1, cat2)]
            if (cat2, cat1) in weighted_relationships:
                matrix[i][j] = weighted_relationships[(cat2, cat1)]

    return matrix, categories


if __name__ == "__main__":
    # Create and visualize basic adjacency matrix
    matrix, categories = create_voc_adjacency_matrix()
    # print(matrix, categories)
    visualize_adjacency_matrix(matrix, categories)
    analyze_matrix(matrix, categories)

    print("\n--- Weighted Relationships ---")
    # Create and visualize weighted relationships
    weighted_matrix, categories = create_weighted_relationships()

    analyze_matrix(weighted_matrix, categories)
    # np.fill_diagonal(weighted_matrix, 1)
    weighted_matrix = preprocess_adj(matrix, len(weighted_matrix))
    print(weighted_matrix)
    visualize_adjacency_matrix(weighted_matrix, categories)
    np.save('VOC2017.npy', weighted_matrix)