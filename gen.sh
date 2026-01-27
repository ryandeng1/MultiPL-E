#!/bin/bash

set -e

OUTPUT_DIR=solutions/gpt_oss_20b_base
MODEL=openai/gpt-oss-20b
LANG=cpp

uv run generate_solutions.py --output_dir ${OUTPUT_DIR} --model ${MODEL} --lang ${LANG} --root_dataset humaneval --num_completions 3
uv run generate_solutions.py --output_dir ${OUTPUT_DIR} --model ${MODEL} --lang ${LANG} --root_dataset mbpp --num_completions 3

