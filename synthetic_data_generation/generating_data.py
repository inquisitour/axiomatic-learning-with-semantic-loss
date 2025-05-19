import random
import string
import json

def generate_node_name(length=3):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_chain(length=3, flip_prob=0.5):
    nodes = [generate_node_name(random.randint(1, 3)) for _ in range(length)]
    edges = []

    for i in range(len(nodes) - 1):
        if random.random() < flip_prob:
            edges.append((nodes[i+1], nodes[i]))  # flipped edge
        else:
            edges.append((nodes[i], nodes[i+1]))  # normal direction
    return nodes, edges

def build_premise(edges):
    return ' '.join([f"{a} causes {b}." for a, b in edges])

def find_path(edges, start, end, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for a, b in edges:
        if a == start and b not in visited:
            if b == end or find_path(edges, b, end, visited):
                return True
    return False

def generate_example():
    chain_len = random.randint(3, 6)
    nodes, edges = generate_chain(chain_len, flip_prob=0.3)
    
    # Pick a random pair to query
    i, j = sorted(random.sample(range(len(nodes)), 2))
    start_node = nodes[i]
    end_node = nodes[j]

    premise = build_premise(edges)
    hypothesis = f"Does {start_node} cause {end_node}?"
    
    label = "Yes" if find_path(edges, start_node, end_node) else "No"
    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label
    }

def generate_dataset(n=1000):
    data = [generate_example() for _ in range(n)]
    return data

# Save to JSONL file
if __name__ == "__main__":
    n = 10
    dataset = generate_dataset(n)
    with open("axiomatic_transitivity_data.jsonl", "w") as f:
        for example in dataset:
            f.write(json.dumps(example) + "\n")
    print(f"Generated {n} examples in axiomatic_transitivity_data.jsonl")
