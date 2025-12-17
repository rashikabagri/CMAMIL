import os
import pickle
import torch


def load_pickled_graph(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_graphs_from_directory(base_dir):
    class_dirs = sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    )

    for class_idx, class_dir in enumerate(class_dirs):
        class_path = os.path.join(base_dir, class_dir)
        files = sorted(f for f in os.listdir(class_path) if f.endswith(".pkl"))

        for fname in files:
            graph = load_pickled_graph(os.path.join(class_path, fname))
            features = torch.tensor(
                [graph.nodes[n]["feature"] for n in graph.nodes],
                dtype=torch.float32
            )
            yield features, class_idx


def data_generator(base_5x, base_10x, base_20x):
    gen_5 = load_graphs_from_directory(base_5x)
    gen_10 = load_graphs_from_directory(base_10x)
    gen_20 = load_graphs_from_directory(base_20x)

    for (f5, y5), (f10, y10), (f20, y20) in zip(gen_5, gen_10, gen_20):
        assert y5 == y10 == y20
        yield f5, f10, f20, y5
