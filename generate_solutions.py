import argparse
import json
import re
import time

from datasets import load_dataset
from typing import List
from openai import OpenAI
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, as_completed

def llm_generate(client, model: str, prompt: str) -> str:
    completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            reasoning_effort="low",
            # temperature=0.6,
            # top_p=0.95,
            temperature=1.0,
            top_p=1.0,
        )

    text = completion.choices[0].message.content
    return text

def generate_solutions(problems: List[dict], model: str, num_completions: int) -> dict[str, list[str]]:
    generations = {}
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="ryan123",
    )

    def process_problem(problem: dict, ) -> tuple[str, list[str]]:
        problem_name = problem["name"]
        prompt = problem["prompt"]
        responses = []
        for _ in range(num_completions):
            response = llm_generate(client, model, prompt)
            responses.append(response)
        return problem_name, responses

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_problem, problem) for problem in problems]
        for future in as_completed(futures):
            problem_name, responses = future.result()
            generations[problem_name] = responses

    return generations

def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate solutions to Effibench-X")
    parser.add_argument(
        "--output_dir", 
        required=True,
        type=str,
        help="output file for generations"
    )
    parser.add_argument(
        "--model", 
        required=True,
        type=str,
        help="output file for generations"
    )
    parser.add_argument(
        "--lang", 
        required=True,
        choices=["python3", "cpp"],
        type=str,
        help="output file for generations"
    )
    parser.add_argument(
        "--root_dataset", 
        required=True,
        choices=["humaneval", "mbpp"],
        type=str,
        help="output file for generations"
    )
    parser.add_argument(
        "--num_completions", 
        required=True,
        type=int,
        help="output file for generations"
    )
    args = parser.parse_args()
    DATASET_REVISION = "3a9e8d226c127392ce81dca29d09f33f0ce3247d"
    
    dataset = load_dataset("nuprl/MultiPL-E", f"{args.root_dataset}-{args.lang}", revision=DATASET_REVISION, split="test")
    lst_problems = []
    for obj in dataset:
        lst_problems.append(obj)
    
    generations = generate_solutions(lst_problems, args.model, args.num_completions)

    for problem_name in generations:
        path = Path(args.output_dir) / f"{problem_name}.json"
        with open(path, "w") as f:
            d = {"completions": generations[problem_name]}
            json.dump(d, f)

if __name__ == "__main__":
    main()
