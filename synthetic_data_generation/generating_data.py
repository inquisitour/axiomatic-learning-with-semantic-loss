import random
import string
import json
from itertools import combinations
from collections import defaultdict

class AxiomaticDataGenerator:
    def __init__(self):
        self.node_name_length_train = (1, 3)  # Training node name length range
        self.node_name_length_eval = (8, 10)  # Evaluation node name length range
        self.chain_length_train = (3, 6)      # Training chain length range
        self.chain_length_eval = (7, 15)      # Evaluation chain length range
        self.branching_factor_train = (0.6, 0.8)  # Training branching factor range
        self.branching_factor_eval = (1.4, 2.0)   # Evaluation branching factor range
    
    def generate_node_name(self, length_range):
        """Generate random node names with variable length"""
        length = random.randint(*length_range)
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def generate_sequential_chain(self, length, name_length_range, flip_prob=0.0):
        """Generate a sequential chain with optional edge flipping"""
        nodes = [self.generate_node_name(name_length_range) for _ in range(length)]
        edges = []
        
        for i in range(len(nodes) - 1):
            if random.random() < flip_prob:
                edges.append((nodes[i+1], nodes[i]))  # flipped edge
            else:
                edges.append((nodes[i], nodes[i+1]))  # normal direction
        return nodes, edges
    
    def generate_branched_graph(self, num_nodes, branching_factor, name_length_range):
        """Generate a graph with branching structure using Erdos-Renyi model"""
        nodes = [self.generate_node_name(name_length_range) for _ in range(num_nodes)]
        edges = []
        
        # Ensure the graph remains a DAG
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if random.random() < branching_factor / num_nodes:
                    edges.append((nodes[i], nodes[j]))
        
        # Ensure all nodes are connected (weakly)
        for i in range(num_nodes - 1):
            if not any(a == nodes[i] or b == nodes[i] for a, b in edges):
                edges.append((nodes[i], nodes[i+1]))
        
        return nodes, edges
    
    def build_premise(self, edges):
        """Convert edges to natural language premise"""
        return ' '.join([f"{a} causes {b}." for a, b in edges])
    
    def is_path(self, edges, start, end, visited=None):
        """Check if there's a path from start to end in the DAG"""
        if visited is None:
            visited = set()
        visited.add(start)
        
        for a, b in edges:
            if a == start and b not in visited:
                if b == end or self.is_path(edges, b, end, visited):
                    return True
        return False
    
    def is_d_separated(self, edges, node1, node2, conditioning_set=set()):
        """Check d-separation between two nodes given a conditioning set"""
        # Implement d-separation rules from Pearl's criteria
        # This is a simplified version - full implementation would need more complex path analysis
        all_paths = self.find_all_paths(edges, node1, node2)
        
        for path in all_paths:
            blocked = False
            for i in range(1, len(path)-1):
                node = path[i]
                prev_node = path[i-1]
                next_node = path[i+1]
                
                # Check for chain or fork structure
                if ((prev_node, node) in edges and (node, next_node) in edges) or \
                   ((next_node, node) in edges and (node, prev_node) in edges):
                    # Chain or fork - blocked if node is in conditioning set
                    if node in conditioning_set:
                        blocked = True
                        break
                
                # Check for collider structure
                elif ((prev_node, node) in edges and (next_node, node) in edges):
                    # Collider - blocked if node and its descendants are not in conditioning set
                    if node not in conditioning_set and not self.has_descendant_in_set(edges, node, conditioning_set):
                        blocked = True
                        break
            
            if not blocked:
                return False  # At least one unblocked path exists
        
        return True  # All paths are blocked
    
    def find_all_paths(self, edges, start, end, path=None):
        """Find all paths between two nodes in the DAG"""
        if path is None:
            path = []
        path = path + [start]
        
        if start == end:
            return [path]
        
        paths = []
        for a, b in edges:
            if a == start and b not in path:
                new_paths = self.find_all_paths(edges, b, end, path)
                for p in new_paths:
                    paths.append(p)
        
        return paths
    
    def has_descendant_in_set(self, edges, node, conditioning_set):
        """Check if any descendant of node is in conditioning set"""
        descendants = set()
        queue = [node]
        
        while queue:
            current = queue.pop()
            for a, b in edges:
                if a == current and b not in descendants:
                    if b in conditioning_set:
                        return True
                    descendants.add(b)
                    queue.append(b)
        return False
    
    def generate_transitivity_example(self, chain_length_range, name_length_range, flip_prob=0.0, shuffle=False):
        """Generate a transitivity axiom example"""
        length = random.randint(*chain_length_range)
        nodes, edges = self.generate_sequential_chain(length, name_length_range, flip_prob)
        
        # Pick a random pair to query
        i, j = sorted(random.sample(range(len(nodes)), 2))
        start_node = nodes[i]
        end_node = nodes[j]

        premise = self.build_premise(edges)
        if shuffle:
            premise = self.shuffle_premise(premise)
        
        hypothesis = f"Does {start_node} cause {end_node}?"
        label = "Yes" if self.is_path(edges, start_node, end_node) else "No"
        
        return {
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label
        }
    
    def generate_d_separation_example(self, num_nodes, name_length_range, branching_factor):
        """Generate a d-separation rule example"""
        nodes, edges = self.generate_branched_graph(num_nodes, branching_factor, name_length_range)
        
        # Select two distinct nodes
        node1, node2 = random.sample(nodes, 2)
        
        # Select conditioning set (0 to min(5, num_nodes-2) nodes)
        possible_conditioning_nodes = [n for n in nodes if n != node1 and n != node2]
        conditioning_size = random.randint(0, min(5, len(possible_conditioning_nodes)))
        
        # Handle case where there are no conditioning nodes available
        if possible_conditioning_nodes and conditioning_size > 0:
            conditioning_set = set(random.sample(possible_conditioning_nodes, conditioning_size))
        else:
            conditioning_set = set()
        
        premise = self.build_premise(edges)
        if conditioning_set:
            hypothesis = f"Are {node1} and {node2} d-separated given {{{', '.join(sorted(conditioning_set))}}}?"
        else:
            hypothesis = f"Are {node1} and {node2} d-separated?"
        
        label = "Yes" if self.is_d_separated(edges, node1, node2, conditioning_set) else "No"
        
        return {
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label
        }
    
    def shuffle_premise(self, premise):
        """Shuffle the order of causal statements in premise"""
        statements = [s.strip() for s in premise.split('.') if s.strip()]
        random.shuffle(statements)
        return ' '.join(statements) + ' '  # Add space to match original format
    
    def generate_training_data(self, num_examples, task="transitivity"):
        """Generate training data for specified task"""
        data = []
        
        for _ in range(num_examples):
            if task == "transitivity":
                # For transitivity, use simple chains with some random flipping
                flip_prob = random.choice([0.0, 0.5])  # Either pure chain or some flipped edges
                example = self.generate_transitivity_example(
                    self.chain_length_train,
                    self.node_name_length_train,
                    flip_prob
                )
            elif task == "d-separation":
                # For d-separation, use branched graphs
                # Ensure we have enough nodes for conditioning sets
                length = random.randint(max(3, self.chain_length_train[0]), self.chain_length_train[1])
                branching_factor = random.uniform(*self.branching_factor_train)
                example = self.generate_d_separation_example(
                    length,
                    self.node_name_length_train,
                    branching_factor
                )
            data.append(example)
        
        return data
    
    def generate_evaluation_data(self, num_examples, eval_type="length"):
        """Generate evaluation data of specified type"""
        data = []
        
        for _ in range(num_examples):
            if eval_type == "length":
                # Longer chains than training
                example = self.generate_transitivity_example(
                    self.chain_length_eval,
                    self.node_name_length_train,
                    flip_prob=0.0  # Pure sequential for length generalization
                )
            elif eval_type == "reversed":
                # Completely reversed chains
                nodes, edges = self.generate_sequential_chain(
                    random.randint(*self.chain_length_eval),
                    self.node_name_length_train
                )
                # Reverse all edges
                edges = [(b, a) for a, b in edges]
                # Pick random pair to query
                i, j = sorted(random.sample(range(len(nodes)), 2))
                premise = self.build_premise(edges)
                hypothesis = f"Does {nodes[i]} cause {nodes[j]}?"
                label = "Yes" if self.is_path(edges, nodes[i], nodes[j]) else "No"
                example = {"premise": premise, "hypothesis": hypothesis, "label": label}
            
            elif eval_type == "shuffled":
                # Shuffled order of statements
                example = self.generate_transitivity_example(
                    self.chain_length_eval,
                    self.node_name_length_train,
                    flip_prob=0.5,
                    shuffle=True
                )
            
            elif eval_type == "long_names":
                # Longer node names
                example = self.generate_transitivity_example(
                    self.chain_length_train,  # Same length as training
                    self.node_name_length_eval,
                    flip_prob=0.5
                )
            
            elif eval_type == "branching":
                # Branched graphs for evaluation
                branching_factor = random.uniform(*self.branching_factor_eval)
                length = random.randint(*self.chain_length_eval)
                example = self.generate_d_separation_example(
                    length,
                    self.node_name_length_train,
                    branching_factor
                )
            
            data.append(example)
        
        return data

def save_to_jsonl(data, filename):
    """Save generated data to JSONL file"""
    with open(filename, "w") as f:
        for example in data:
            f.write(json.dumps(example) + "\n")

if __name__ == "__main__":
    generator = AxiomaticDataGenerator()
    
    # Generate training data
    print("Generating training data...")
    transitivity_train = generator.generate_training_data(1000, "transitivity")
    dsep_train = generator.generate_training_data(1000, "d-separation")
    
    save_to_jsonl(transitivity_train, "transitivity_train.jsonl")
    save_to_jsonl(dsep_train, "dsep_train.jsonl")
    
    # Generate evaluation data
    print("Generating evaluation data...")
    length_eval = generator.generate_evaluation_data(200, "length")
    reversed_eval = generator.generate_evaluation_data(200, "reversed")
    shuffled_eval = generator.generate_evaluation_data(200, "shuffled")
    long_names_eval = generator.generate_evaluation_data(200, "long_names")
    branching_eval = generator.generate_evaluation_data(200, "branching")
    
    save_to_jsonl(length_eval, "length_eval.jsonl")
    save_to_jsonl(reversed_eval, "reversed_eval.jsonl")
    save_to_jsonl(shuffled_eval, "shuffled_eval.jsonl")
    save_to_jsonl(long_names_eval, "long_names_eval.jsonl")
    save_to_jsonl(branching_eval, "branching_eval.jsonl")
    
    print("Data generation complete!")