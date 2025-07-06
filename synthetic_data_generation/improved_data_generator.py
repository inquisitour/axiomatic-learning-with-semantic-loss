import random
import string
import json
import logging
from typing import List, Tuple, Set, Dict, Optional, Union
from dataclasses import dataclass
from collections import defaultdict, deque
from itertools import combinations
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for data generation parameters"""
    node_name_length_train: Tuple[int, int] = (1, 3)
    node_name_length_eval: Tuple[int, int] = (8, 10)
    chain_length_train: Tuple[int, int] = (3, 6)
    chain_length_eval: Tuple[int, int] = (7, 15)
    branching_factor_train: Tuple[float, float] = (0.3, 0.6)
    branching_factor_eval: Tuple[float, float] = (0.7, 1.2)
    max_conditioning_set_size: int = 3
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.node_name_length_train[0] <= 0 or self.node_name_length_train[1] <= 0:
            raise ValueError("Node name lengths must be positive")
        if self.chain_length_train[0] < 2 or self.chain_length_train[1] < 2:
            raise ValueError("Chain length must be at least 2")


class GraphGenerator:
    """Handles graph generation with proper DAG constraints"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
    
    def generate_node_name(self, length_range: Tuple[int, int]) -> str:
        """Generate random node names with variable length"""
        try:
            length = random.randint(*length_range)
            if length <= 0:
                raise ValueError("Node name length must be positive")
            return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        except Exception as e:
            logger.error(f"Error generating node name: {e}")
            raise
    
    def generate_sequential_chain(
        self, 
        length: int, 
        name_length_range: Tuple[int, int], 
        flip_prob: float = 0.0
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Generate a sequential chain with optional edge flipping"""
        if length < 2:
            raise ValueError("Chain length must be at least 2")
        
        nodes = [self.generate_node_name(name_length_range) for _ in range(length)]
        edges = []
        
        for i in range(len(nodes) - 1):
            if random.random() < flip_prob:
                edges.append((nodes[i+1], nodes[i]))  # flipped edge
            else:
                edges.append((nodes[i], nodes[i+1]))  # normal direction
        
        return nodes, edges
    
    def generate_dag(
        self, 
        num_nodes: int, 
        edge_density: float, 
        name_length_range: Tuple[int, int]
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Generate a proper DAG with guaranteed topological ordering"""
        if num_nodes < 2:
            raise ValueError("Number of nodes must be at least 2")
        
        nodes = [self.generate_node_name(name_length_range) for _ in range(num_nodes)]
        edges = []
        
        # Create topological ordering to ensure DAG property
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < edge_density:
                    edges.append((nodes[i], nodes[j]))
        
        # Ensure connectivity - add backbone chain if graph is too sparse
        if len(edges) < num_nodes - 1:
            for i in range(num_nodes - 1):
                if not any(nodes[i] in edge for edge in edges):
                    edges.append((nodes[i], nodes[i + 1]))
        
        return nodes, edges


class CausalReasoner:
    """Handles causal reasoning tasks and d-separation"""
    
    def __init__(self):
        self.graph_cache = {}
    
    def build_adjacency_lists(self, edges: List[Tuple[str, str]]) -> Dict[str, Dict[str, List[str]]]:
        """Build adjacency lists for directed graph"""
        graph = {
            'children': defaultdict(list),
            'parents': defaultdict(list),
            'neighbors': defaultdict(set)
        }
        
        for parent, child in edges:
            graph['children'][parent].append(child)
            graph['parents'][child].append(parent)
            graph['neighbors'][parent].add(child)
            graph['neighbors'][child].add(parent)
        
        return graph
    
    def find_path_dfs(
        self, 
        edges: List[Tuple[str, str]], 
        start: str, 
        end: str, 
        visited: Optional[Set[str]] = None
    ) -> bool:
        """Check if there's a directed path from start to end using DFS"""
        if visited is None:
            visited = set()
        
        if start == end:
            return True
        
        if start in visited:
            return False
        
        visited.add(start)
        
        for parent, child in edges:
            if parent == start and child not in visited:
                if self.find_path_dfs(edges, child, end, visited.copy()):
                    return True
        
        return False
    
    def find_all_undirected_paths(
        self, 
        graph: Dict[str, Dict[str, List[str]]], 
        start: str, 
        end: str, 
        max_length: int = 10
    ) -> List[List[str]]:
        """Find all undirected paths between two nodes (for d-separation)"""
        if start == end:
            return [[start]]
        
        paths = []
        queue = deque([(start, [start])])
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_length:
                continue
            
            if current == end and len(path) > 1:
                paths.append(path)
                continue
            
            for neighbor in graph['neighbors'][current]:
                if neighbor not in path:  # Avoid cycles
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
        
        return paths
    
    def is_collider(self, graph: Dict[str, Dict[str, List[str]]], node: str, prev_node: str, next_node: str) -> bool:
        """Check if node is a collider in the path prev_node -> node <- next_node"""
        return (prev_node in graph['parents'][node] and 
                next_node in graph['parents'][node])
    
    def is_chain_or_fork(self, graph: Dict[str, Dict[str, List[str]]], node: str, prev_node: str, next_node: str) -> bool:
        """Check if node is part of a chain or fork"""
        # Chain: prev -> node -> next
        chain = (prev_node in graph['parents'][node] and 
                next_node in graph['children'][node])
        
        # Fork: prev <- node -> next
        fork = (prev_node in graph['children'][node] and 
               next_node in graph['children'][node])
        
        return chain or fork
    
    def has_descendant_in_conditioning_set(
        self, 
        graph: Dict[str, Dict[str, List[str]]], 
        node: str, 
        conditioning_set: Set[str]
    ) -> bool:
        """Check if node has any descendant in the conditioning set"""
        visited = set()
        queue = deque([node])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            for child in graph['children'][current]:
                if child in conditioning_set:
                    return True
                if child not in visited:
                    queue.append(child)
        
        return False
    
    def is_d_separated(
        self, 
        edges: List[Tuple[str, str]], 
        node1: str, 
        node2: str, 
        conditioning_set: Set[str] = None
    ) -> bool:
        """Check d-separation between two nodes given a conditioning set"""
        if conditioning_set is None:
            conditioning_set = set()
        
        if node1 == node2:
            return False
        
        graph = self.build_adjacency_lists(edges)
        paths = self.find_all_undirected_paths(graph, node1, node2)
        
        if not paths:
            return True  # No paths means d-separated
        
        # Check if all paths are blocked
        for path in paths:
            if not self.is_path_blocked(graph, path, conditioning_set):
                return False  # Found an unblocked path
        
        return True  # All paths are blocked
    
    def is_path_blocked(
        self, 
        graph: Dict[str, Dict[str, List[str]]], 
        path: List[str], 
        conditioning_set: Set[str]
    ) -> bool:
        """Check if a path is blocked by the conditioning set"""
        if len(path) < 3:
            return False
        
        for i in range(1, len(path) - 1):
            node = path[i]
            prev_node = path[i - 1]
            next_node = path[i + 1]
            
            if self.is_collider(graph, node, prev_node, next_node):
                # Collider: blocked unless node or its descendant is in conditioning set
                if (node not in conditioning_set and 
                    not self.has_descendant_in_conditioning_set(graph, node, conditioning_set)):
                    return True
            elif self.is_chain_or_fork(graph, node, prev_node, next_node):
                # Chain or fork: blocked if node is in conditioning set
                if node in conditioning_set:
                    return True
        
        return False


class DataValidator:
    """Validates generated examples for correctness"""
    
    def __init__(self, reasoner: CausalReasoner):
        self.reasoner = reasoner
    
    def validate_transitivity_example(self, example: Dict) -> bool:
        """Validate that transitivity example is logically correct"""
        try:
            premise = example['premise']
            hypothesis = example['hypothesis']
            label = example['label']
            
            # Parse edges from premise
            edges = self.parse_premise(premise)
            
            # Extract query nodes from hypothesis
            start_node, end_node = self.parse_hypothesis(hypothesis)
            
            # Check if label matches actual causal structure
            actual_causation = self.reasoner.find_path_dfs(edges, start_node, end_node)
            expected_label = "Yes" if actual_causation else "No"
            
            return label == expected_label
            
        except Exception as e:
            logger.error(f"Error validating transitivity example: {e}")
            return False
    
    def validate_d_separation_example(self, example: Dict) -> bool:
        """Validate that d-separation example is logically correct"""
        try:
            premise = example['premise']
            hypothesis = example['hypothesis']
            label = example['label']
            
            # Parse edges from premise
            edges = self.parse_premise(premise)
            
            # Extract query information from hypothesis
            node1, node2, conditioning_set = self.parse_d_separation_hypothesis(hypothesis)
            
            # Check if label matches actual d-separation
            actual_d_separation = self.reasoner.is_d_separated(edges, node1, node2, conditioning_set)
            expected_label = "Yes" if actual_d_separation else "No"
            
            return label == expected_label
            
        except Exception as e:
            logger.error(f"Error validating d-separation example: {e}")
            return False
    
    def parse_premise(self, premise: str) -> List[Tuple[str, str]]:
        """Parse premise text to extract causal edges"""
        edges = []
        statements = [s.strip() for s in premise.split('.') if s.strip()]
        
        for statement in statements:
            if ' causes ' in statement:
                parts = statement.split(' causes ')
                if len(parts) == 2:
                    edges.append((parts[0].strip(), parts[1].strip()))
        
        return edges
    
    def parse_hypothesis(self, hypothesis: str) -> Tuple[str, str]:
        """Parse hypothesis to extract start and end nodes"""
        # Extract from "Does X cause Y?"
        if hypothesis.startswith("Does ") and hypothesis.endswith("?"):
            content = hypothesis[5:-1]  # Remove "Does " and "?"
            if ' cause ' in content:
                parts = content.split(' cause ')
                if len(parts) == 2:
                    return parts[0].strip(), parts[1].strip()
        
        raise ValueError(f"Cannot parse hypothesis: {hypothesis}")
    
    def parse_d_separation_hypothesis(self, hypothesis: str) -> Tuple[str, str, Set[str]]:
        """Parse d-separation hypothesis to extract nodes and conditioning set"""
        # Extract from "Are X and Y d-separated given {Z1, Z2}?"
        if "d-separated" in hypothesis:
            if "given" in hypothesis:
                # Has conditioning set
                parts = hypothesis.split("given")
                nodes_part = parts[0].strip()
                conditioning_part = parts[1].strip()
                
                # Extract nodes
                nodes_match = nodes_part.replace("Are ", "").replace(" d-separated", "")
                if " and " in nodes_match:
                    node1, node2 = [n.strip() for n in nodes_match.split(" and ")]
                
                # Extract conditioning set
                conditioning_set = set()
                if "{" in conditioning_part and "}" in conditioning_part:
                    conditioning_str = conditioning_part.split("{")[1].split("}")[0]
                    conditioning_set = {n.strip() for n in conditioning_str.split(",")}
                
                return node1, node2, conditioning_set
            else:
                # No conditioning set
                nodes_match = hypothesis.replace("Are ", "").replace(" d-separated?", "")
                if " and " in nodes_match:
                    node1, node2 = [n.strip() for n in nodes_match.split(" and ")]
                return node1, node2, set()
        
        raise ValueError(f"Cannot parse d-separation hypothesis: {hypothesis}")


class AxiomaticDataGenerator:
    """Main class for generating axiomatic reasoning data"""
    
    def __init__(self, config: GenerationConfig = None):
        self.config = config or GenerationConfig()
        self.graph_generator = GraphGenerator(self.config)
        self.reasoner = CausalReasoner()
        self.validator = DataValidator(self.reasoner)
    
    def build_premise(self, edges: List[Tuple[str, str]]) -> str:
        """Convert edges to natural language premise"""
        return ' '.join([f"{a} causes {b}." for a, b in edges])
    
    def shuffle_premise(self, premise: str) -> str:
        """Shuffle the order of causal statements in premise"""
        statements = [s.strip() for s in premise.split('.') if s.strip()]
        random.shuffle(statements)
        return ' '.join(statements) + '.'
    
    def generate_transitivity_example(
        self, 
        chain_length_range: Tuple[int, int], 
        name_length_range: Tuple[int, int], 
        flip_prob: float = 0.0, 
        shuffle: bool = False
    ) -> Dict:
        """Generate a transitivity axiom example"""
        max_attempts = 10
        
        for attempt in range(max_attempts):
            try:
                length = random.randint(*chain_length_range)
                nodes, edges = self.graph_generator.generate_sequential_chain(
                    length, name_length_range, flip_prob
                )
                
                # Pick a random pair to query
                if len(nodes) < 2:
                    continue
                
                i, j = sorted(random.sample(range(len(nodes)), 2))
                start_node = nodes[i]
                end_node = nodes[j]
                
                premise = self.build_premise(edges)
                if shuffle:
                    premise = self.shuffle_premise(premise)
                
                hypothesis = f"Does {start_node} cause {end_node}?"
                label = "Yes" if self.reasoner.find_path_dfs(edges, start_node, end_node) else "No"
                
                example = {
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "label": label
                }
                
                # Validate the example
                if self.validator.validate_transitivity_example(example):
                    return example
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                continue
        
        raise RuntimeError("Failed to generate valid transitivity example after maximum attempts")
    
    def generate_d_separation_example(
        self, 
        num_nodes: int, 
        name_length_range: Tuple[int, int], 
        edge_density: float
    ) -> Dict:
        """Generate a d-separation rule example"""
        max_attempts = 10
        
        for attempt in range(max_attempts):
            try:
                nodes, edges = self.graph_generator.generate_dag(
                    num_nodes, edge_density, name_length_range
                )
                
                if len(nodes) < 2:
                    continue
                
                # Select two distinct nodes
                node1, node2 = random.sample(nodes, 2)
                
                # Select conditioning set
                possible_conditioning_nodes = [n for n in nodes if n != node1 and n != node2]
                conditioning_size = random.randint(0, min(self.config.max_conditioning_set_size, len(possible_conditioning_nodes)))
                
                conditioning_set = set()
                if possible_conditioning_nodes and conditioning_size > 0:
                    conditioning_set = set(random.sample(possible_conditioning_nodes, conditioning_size))
                
                premise = self.build_premise(edges)
                
                if conditioning_set:
                    hypothesis = f"Are {node1} and {node2} d-separated given {{{', '.join(sorted(conditioning_set))}}}?"
                else:
                    hypothesis = f"Are {node1} and {node2} d-separated?"
                
                label = "Yes" if self.reasoner.is_d_separated(edges, node1, node2, conditioning_set) else "No"
                
                example = {
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "label": label
                }
                
                # Validate the example
                if self.validator.validate_d_separation_example(example):
                    return example
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                continue
        
        raise RuntimeError("Failed to generate valid d-separation example after maximum attempts")
    
    def generate_training_data(self, num_examples: int, task: str = "transitivity") -> List[Dict]:
        """Generate training data for specified task"""
        data = []
        logger.info(f"Generating {num_examples} training examples for {task}")
        
        for i in range(num_examples):
            try:
                if task == "transitivity":
                    flip_prob = random.choice([0.0, 0.3, 0.5])
                    example = self.generate_transitivity_example(
                        self.config.chain_length_train,
                        self.config.node_name_length_train,
                        flip_prob
                    )
                elif task == "d-separation":
                    length = random.randint(
                        max(3, self.config.chain_length_train[0]), 
                        self.config.chain_length_train[1]
                    )
                    edge_density = random.uniform(*self.config.branching_factor_train)
                    example = self.generate_d_separation_example(
                        length,
                        self.config.node_name_length_train,
                        edge_density
                    )
                else:
                    raise ValueError(f"Unknown task: {task}")
                
                data.append(example)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Generated {i + 1}/{num_examples} examples")
                    
            except Exception as e:
                logger.error(f"Error generating example {i + 1}: {e}")
                continue
        
        logger.info(f"Successfully generated {len(data)} examples")
        return data
    
    def generate_evaluation_data(self, num_examples: int, eval_type: str = "length") -> List[Dict]:
        """Generate evaluation data of specified type"""
        data = []
        logger.info(f"Generating {num_examples} evaluation examples for {eval_type}")
        
        for i in range(num_examples):
            try:
                if eval_type == "length":
                    example = self.generate_transitivity_example(
                        self.config.chain_length_eval,
                        self.config.node_name_length_train,
                        flip_prob=0.0
                    )
                elif eval_type == "reversed":
                    nodes, edges = self.graph_generator.generate_sequential_chain(
                        random.randint(*self.config.chain_length_eval),
                        self.config.node_name_length_train
                    )
                    # Reverse all edges
                    edges = [(b, a) for a, b in edges]
                    i, j = sorted(random.sample(range(len(nodes)), 2))
                    premise = self.build_premise(edges)
                    hypothesis = f"Does {nodes[i]} cause {nodes[j]}?"
                    label = "Yes" if self.reasoner.find_path_dfs(edges, nodes[i], nodes[j]) else "No"
                    example = {"premise": premise, "hypothesis": hypothesis, "label": label}
                    
                elif eval_type == "shuffled":
                    example = self.generate_transitivity_example(
                        self.config.chain_length_eval,
                        self.config.node_name_length_train,
                        flip_prob=0.5,
                        shuffle=True
                    )
                elif eval_type == "long_names":
                    example = self.generate_transitivity_example(
                        self.config.chain_length_train,
                        self.config.node_name_length_eval,
                        flip_prob=0.5
                    )
                elif eval_type == "branching":
                    edge_density = random.uniform(*self.config.branching_factor_eval)
                    length = random.randint(*self.config.chain_length_eval)
                    example = self.generate_d_separation_example(
                        length,
                        self.config.node_name_length_train,
                        edge_density
                    )
                else:
                    raise ValueError(f"Unknown evaluation type: {eval_type}")
                
                data.append(example)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Generated {i + 1}/{num_examples} examples")
                    
            except Exception as e:
                logger.error(f"Error generating example {i + 1}: {e}")
                continue
        
        logger.info(f"Successfully generated {len(data)} examples")
        return data


def save_to_jsonl(data: List[Dict], filename: str) -> None:
    """Save generated data to JSONL file"""
    try:
        with open(filename, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")
        logger.info(f"Saved {len(data)} examples to {filename}")
    except Exception as e:
        logger.error(f"Error saving to {filename}: {e}")
        raise


def main():
    """Main function to generate all datasets"""
    config = GenerationConfig()
    generator = AxiomaticDataGenerator(config)
    
    try:
        # Generate training data
        logger.info("=== Starting Training Data Generation ===")
        transitivity_train = generator.generate_training_data(1000, "transitivity")
        dsep_train = generator.generate_training_data(1000, "d-separation")
        
        save_to_jsonl(transitivity_train, "transitivity_train.jsonl")
        save_to_jsonl(dsep_train, "dsep_train.jsonl")
        
        # Generate evaluation data
        logger.info("=== Starting Evaluation Data Generation ===")
        evaluation_types = ["length", "reversed", "shuffled", "long_names", "branching"]
        
        for eval_type in evaluation_types:
            eval_data = generator.generate_evaluation_data(200, eval_type)
            save_to_jsonl(eval_data, f"{eval_type}_eval.jsonl")
        
        logger.info("=== Data Generation Complete ===")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
