import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp


def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix: D^(-1/2) A D^(-1/2)
    """
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return normalized_adj


def preprocess_adj(adj_matrix, num_nodes):
    """
    Preprocessing of adjacency matrix for GCN
    """
    adj = adj_matrix + np.eye(num_nodes)
    adj_sparse = sp.csr_matrix(adj)
    adj_normalized = normalize_adj(adj_sparse)
    adj_normalized_dense = adj_normalized.toarray()
    return adj_normalized_dense



def create_coco_adjacency_matrix():
    categories = [
        'CLS','person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    matrix = np.zeros((len(categories), len(categories)))

    weighted_relationships = {
        # Person interactions with objects
        ('person', 'bicycle'): 0.8,
        ('person', 'car'): 0.7,
        ('person', 'motorcycle'): 0.7,
        ('person', 'cell phone'): 0.9,
        ('person', 'laptop'): 0.8,
        ('person', 'chair'): 0.9,
        ('person', 'couch'): 0.8,
        ('person', 'bed'): 0.7,
        ('person', 'book'): 0.7,
        ('person', 'tv'): 0.7,
        ('person', 'dining table'): 0.8,
        ('person', 'backpack'): 0.8,
        ('person', 'umbrella'): 0.7,
        ('person', 'handbag'): 0.7,
        ('person', 'tie'): 0.8,
        ('person', 'skateboard'): 0.7,
        ('person', 'surfboard'): 0.7,
        ('person', 'tennis racket'): 0.7,
        ('person', 'bottle'): 0.6,
        ('person', 'wine glass'): 0.6,
        ('person', 'cup'): 0.7,
        ('person', 'fork'): 0.6,
        ('person', 'knife'): 0.6,
        ('person', 'spoon'): 0.6,

        # Kitchen/Dining items
        ('fork', 'knife'): 0.9,
        ('fork', 'spoon'): 0.9,
        ('knife', 'spoon'): 0.9,
        ('cup', 'wine glass'): 0.7,
        ('bowl', 'spoon'): 0.8,
        ('bowl', 'fork'): 0.7,
        ('bowl', 'knife'): 0.7,
        ('dining table', 'chair'): 0.9,
        ('dining table', 'wine glass'): 0.7,
        ('dining table', 'cup'): 0.8,
        ('dining table', 'fork'): 0.8,
        ('dining table', 'knife'): 0.8,
        ('dining table', 'spoon'): 0.8,
        ('dining table', 'bowl'): 0.8,

        # Food items and their containers/utensils
        ('pizza', 'plate'): 0.8,
        ('sandwich', 'plate'): 0.8,
        ('hot dog', 'plate'): 0.8,
        ('cake', 'plate'): 0.8,
        ('donut', 'plate'): 0.7,
        ('apple', 'bowl'): 0.6,
        ('orange', 'bowl'): 0.6,
        ('banana', 'bowl'): 0.6,
        ('broccoli', 'bowl'): 0.7,
        ('carrot', 'bowl'): 0.7,

        # Furniture relationships
        ('chair', 'dining table'): 0.9,
        ('couch', 'tv'): 0.8,
        ('bed', 'pillow'): 0.9,
        ('chair', 'desk'): 0.8,
        ('couch', 'coffee table'): 0.8,
        ('bed', 'teddy bear'): 0.6,

        # Electronics relationships
        ('tv', 'remote'): 0.9,
        ('laptop', 'mouse'): 0.9,
        ('laptop', 'keyboard'): 0.9,
        ('tv', 'game console'): 0.7,
        ('laptop', 'desk'): 0.8,
        ('cell phone', 'charger'): 0.8,

        # Animals and their environment
        ('dog', 'cat'): 0.6,
        ('horse', 'zebra'): 0.5,
        ('cow', 'sheep'): 0.6,
        ('bird', 'potted plant'): 0.4,
        ('elephant', 'giraffe'): 0.5,
        ('dog', 'teddy bear'): 0.4,
        ('cat', 'couch'): 0.6,
        ('dog', 'couch'): 0.6,

        # Sports/Recreation equipment
        ('baseball bat', 'baseball glove'): 0.9,
        ('tennis racket', 'sports ball'): 0.7,
        ('skis', 'snowboard'): 0.7,
        ('frisbee', 'sports ball'): 0.6,
        ('skateboard', 'sports ball'): 0.5,
        ('surfboard', 'sports ball'): 0.5,

        # Transportation related
        ('car', 'truck'): 0.7,
        ('motorcycle', 'bicycle'): 0.6,
        ('train', 'bus'): 0.6,
        ('car', 'traffic light'): 0.8,
        ('car', 'stop sign'): 0.8,
        ('car', 'parking meter'): 0.7,
        ('bus', 'traffic light'): 0.8,
        ('bus', 'stop sign'): 0.8,
        ('motorcycle', 'traffic light'): 0.7,
        ('bicycle', 'traffic light'): 0.7,

        # Kitchen appliances
        ('refrigerator', 'microwave'): 0.7,
        ('refrigerator', 'oven'): 0.7,
        ('sink', 'microwave'): 0.6,
        ('sink', 'oven'): 0.6,
        ('sink', 'toaster'): 0.6,
        ('sink', 'refrigerator'): 0.7,

        # Bathroom items
        ('sink', 'toothbrush'): 0.8,
        ('sink', 'hair drier'): 0.7,
        ('toilet', 'sink'): 0.8,

        # Personal items
        ('backpack', 'book'): 0.7,
        ('handbag', 'cell phone'): 0.7,
        ('suitcase', 'handbag'): 0.6,
        ('tie', 'suit'): 0.8,

        # Indoor decorative items
        ('vase', 'potted plant'): 0.7,
        ('clock', 'wall'): 0.8,
        ('tv', 'wall'): 0.7,

        # Outdoor relationships
        ('bench', 'potted plant'): 0.5,
        ('fire hydrant', 'stop sign'): 0.4,
        ('traffic light', 'stop sign'): 0.6,
        ('parking meter', 'stop sign'): 0.5,

        # Weather related
        ('umbrella', 'rain'): 0.8,
        ('boat', 'water'): 0.9,
        ('surfboard', 'water'): 0.9
    }

    # Fill the matrix
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if (cat1, cat2) in weighted_relationships:
                matrix[i][j] = weighted_relationships[(cat1, cat2)]
            if (cat2, cat1) in weighted_relationships:
                matrix[i][j] = weighted_relationships[(cat2, cat1)]

    return matrix, categories




def visualize_adjacency_matrix(matrix, categories):
    plt.figure(figsize=(20, 16))
    df = pd.DataFrame(matrix, index=categories, columns=categories)
    sns.heatmap(df, annot=True, cmap='YlOrRd', fmt='.1f',
                square=True, cbar_kws={'label': 'Relationship Strength'})
    plt.title('COCO Categories Adjacency Matrix', pad=20)
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def analyze_matrix(matrix, categories):
    print("Matrix Analysis:")
    print(f"Total relationships: {np.sum(matrix)}")
    print(f"Network density: {np.sum(matrix) / (len(categories) * len(categories)):.3f}")

    connections = np.sum(matrix, axis=1)
    category_connections = [(cat, conn) for cat, conn in zip(categories, connections)]
    print("\nMost connected categories:")
    for cat, conn in sorted(category_connections, key=lambda x: x[1], reverse=True)[:5]:
        print(f"{cat}: {conn:.2f} connections")


if __name__ == "__main__":
    # Create and analyze adjacency matrix
    matrix, categories = create_coco_adjacency_matrix()

    # Analyze original matrix
    print("Original Matrix Analysis:")
    analyze_matrix(matrix, categories)

    # Preprocess and normalize
    processed_matrix = preprocess_adj(matrix, len(categories))

    # Analyze processed matrix
    print("\nProcessed Matrix Analysis:")
    analyze_matrix(processed_matrix, categories)

    # Visualize
    visualize_adjacency_matrix(processed_matrix, categories)

    # Save matrix
    np.save('COCO_adjacency.npy', processed_matrix)