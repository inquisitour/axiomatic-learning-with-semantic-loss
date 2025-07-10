# new_data.py
from improved_data_generator import AxiomaticDataGenerator, GenerationConfig  
import json

def generate_branching_evaluation_data(num_examples: int, output_file: str):
    """
    Generate branching evaluation dataset and save to JSONL file.
    
    Args:
        num_examples: Number of examples to generate
        output_file: Path to output JSONL file
    """
    # Initialize the generator with default config
    config = GenerationConfig()
    generator = AxiomaticDataGenerator(config)
    
    # Generate branching evaluation data
    branching_data = generator.generate_evaluation_data(num_examples, eval_type="branching")
    
    # Save to JSONL file
    with open(output_file, 'w') as f:
        for example in branching_data:
            f.write(json.dumps(example) + '\n')
    
    print(f"Successfully generated {len(branching_data)} branching evaluation examples to {output_file}")

if __name__ == "__main__":
    # Example usage
    generate_branching_evaluation_data(
        num_examples=2000, 
        output_file="branching_eval.jsonl"
    )