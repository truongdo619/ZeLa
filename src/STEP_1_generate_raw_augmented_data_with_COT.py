import argparse
import json
import os
import random
import re
import sys
import torch

from tqdm import tqdm
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer
)

# Local utility functions, presumably defined in utils.py
# (Make sure you have get_node_info, Tree, and write_json in utils.py or adapt as needed)
from utils import get_node_info, Tree, write_json

################################################################################
# Argument parsing
################################################################################

def parse_arguments():
    """
    Parse and return command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Process translations with step-by-step generation.')
    parser.add_argument('--model', type=str, default="meta-llama/Llama-2-70b-hf",
                        help='Model name or path')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for generation')
    parser.add_argument('--max_new_tokens', type=int, default=40,
                        help='Max new tokens for generation')
    parser.add_argument('--save_step', type=int, default=100,
                        help='Number of steps between saving progress')
    parser.add_argument('--google_translated', action='store_true',
                        help='Flag to indicate if using Google-translated data')
    args, _ = parser.parse_known_args()
    return args

################################################################################
# I/O Utilities
################################################################################

def save_to_jsonl(data_list, filename):
    """
    Save a list of dict items to a JSON lines file.
    """
    with open(filename, 'w', encoding='utf-8') as jsonl_file:
        for item in data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            jsonl_file.write(json_line + '\n')


def read_jsonl_file(file_path):
    """
    Read a JSON lines file and return a list of Python dicts.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

################################################################################
# Model loading
################################################################################

def load_model_and_tokenizer(model_name):
    """
    Load the specified model and tokenizer from Hugging Face.
    Return (tokenizer, model).
    """
    # For device-map usage:
    device_map = "auto"
    
    if model_name == "google/flan-t5-xxl":
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16
        )
    elif model_name == "google/flan-ul2":
        model_id = "google/flan-ul2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = T5ForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.float16
        )
    elif model_name == "tiiuae/falcon-40b-instruct":
        model_id = "tiiuae/falcon-40b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device_map
        )
    elif model_name == "huggyllama/llama-65b":
        tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-65b")
        model = LlamaForCausalLM.from_pretrained(
            "huggyllama/llama-65b",
            device_map=device_map,
            load_in_8bit=True
        )
    elif model_name == "huggyllama/llama-7b":
        tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
        model = LlamaForCausalLM.from_pretrained(
            "huggyllama/llama-7b",
            device_map=device_map
        )
    elif model_name == "huggyllama/llama-13b":
        tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-13b")
        model = LlamaForCausalLM.from_pretrained(
            "huggyllama/llama-13b",
            device_map=device_map
        )
    elif model_name == "meta-llama/Llama-2-70b-hf":
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
        model = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-70b-hf",
            device_map=device_map,
            load_in_8bit=True
        )
    else:
        # Default fallback to general Llama-like model with 8-bit load
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            load_in_8bit=True
        )
    
    model.eval()

    # Compile model if supported (PyTorch 2.0+ and non-Windows)
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return tokenizer, model

################################################################################
# Processing / Generation Helper Functions
################################################################################

def format_output_cot(output_text, input_text):
    """
    Format the chain-of-thought (CoT) style output by removing unwanted tokens
    and splitting steps line by line.
    """
    cleaned_output_text = (
        output_text
        .replace("<s>", "")
        .replace("<pad> ", "")
        .replace("</s>", "")
        .replace(input_text, "")
    )
    output_sample = cleaned_output_text.split("\n\n")[0]
    predicted_steps = output_sample.split("\n")

    result = []
    pattern = r"Step \d+: "  # Regex pattern to match "Step [number]: "
    for step in predicted_steps:
        tmp = re.sub(pattern, "", step).strip()
        if tmp:
            result.append(tmp)
    return result


def build_cot(utterance, decoupled_form, label_types, entity_levels, start_positions, end_positions):
    """
    Build chain-of-thought style annotation. 
    Example: 
    [INTENT <utterance> ]
    [LABEL some tokens ]
    ...
    """
    # Exclude bracket tokens in the decoupled form
    tokens = [
        token for token in decoupled_form.split(" ")
        if not (token.startswith("[") or token.startswith("]"))
    ]

    result_list = [f"[{label_types[0]} {utterance} ]"]
    # For each subsequent label, create bracketed annotation
    for label_type, ent_level, start_pos, end_pos in zip(
        label_types[1:], entity_levels[1:], start_positions[1:], end_positions[1:]
    ):
        snippet = " ".join(tokens[start_pos : end_pos + 1])
        result_list.append(f"[{label_type} {snippet} ]")
    return result_list


def query_model(input_texts, tokenizer, model, max_new_tokens=40, temperature=0.0):
    """
    Run the model on the given list of input strings and return the decoded outputs.
    """
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_current_step(output_text, num_processed_steps):
    """
    Extract the newly generated step from the model output. The line we want 
    is offset by num_processed_steps + 2 (based on how the text is structured).
    """
    cleaned = (
        output_text.replace("<s>", "")
        .replace("<pad> ", "")
        .replace("</s>", "")
    )
    lines = cleaned.split("\n")
    # The model is generating line-by-line steps; adjust indexing if needed
    step_line_index = num_processed_steps + 2

    if step_line_index < len(lines):
        predicted_step = lines[step_line_index]
        # Everything after "|" in that line is the actual new bracket label 
        return predicted_step.split("|")[-1].strip()
    return ""


def build_new_query(current_query, remaining_step_labels, new_output_step, current_llm_outputs):
    """
    Update the query string to include the newly generated step output. 
    If there's another label pending, append a new 'Step' prompt.
    """
    query_tokens = current_query.split(" ")[:-1]  # remove the trailing chunk
    query_so_far = " ".join(query_tokens)

    step_count = len(current_llm_outputs)

    # If there are more labels to generate, produce another "Step x:" prompt
    if len(remaining_step_labels) > 0:
        next_label = remaining_step_labels[0]
        return (
            f"{query_so_far} {new_output_step}\n"
            f"Step {step_count + 1}: {next_label} | {next_label.split(' ')[0]}"
        )
    else:
        # If no remaining labels, just append the final step output
        return f"{query_so_far} {new_output_step}\n"

################################################################################
# Core Logic
################################################################################

FULL_FORM_LANGS = {
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "th": "Thai"
}


def base_query_generate(translated_item, en_item, lang):
    """
    Creates an initial query string for the bilingual example:
      Language: English | <LANGUAGE>
      Input: <utterance_in_en> | <utterance_in_foreign_language>
      Step 1: [INTENT <utterance_in_en>] | [INTENT <utterance_in_foreign_language>]
      ...
    Returns:
      (initial_query_str, remaining_step_labels, current_llm_output)
    """
    str_query = f"Language: English | {FULL_FORM_LANGS[lang]}\n"
    str_query += (
        f"Input: {translated_item['utterance']} | "
        f"{translated_item['translated_utterance']}\n"
        "Step 1: "
    )

    try:
        node_info = get_node_info(Tree(en_item['decoupled_form']))
        label_types = [item[2] for item in node_info]
        entity_levels = [item[1] for item in node_info]
        start_positions = [int(item[0].split(";")[0]) for item in node_info]
        end_positions = [int(item[0].split(";")[1]) for item in node_info]
    except Exception:
        # If decoupled_form is not parseable, just treat it as a single intent
        label_types = [en_item["intent"]]
        entity_levels = []
        start_positions = []
        end_positions = []

    # Build chain-of-thought from the English sample
    cot_steps = build_cot(
        en_item['utterance'],
        en_item['decoupled_form'],
        label_types,
        entity_levels,
        start_positions,
        end_positions
    )

    # Step 1 includes the first bracket from EN and the first bracket from the translated
    str_query += f"{cot_steps[0]} | [{label_types[0]} {translated_item['translated_utterance']} ]\n"

    # If there are more steps to generate, set them aside
    remaining_steps = cot_steps[1:]
    initial_output = f"[{label_types[0]} {translated_item['translated_utterance']} ]"

    if len(remaining_steps) != 0:
        # Manually add part of Step 2 in the query to prime the next bracket
        str_query += f"Step 2: {remaining_steps[0]} | [{remaining_steps[0].split(' ')[0]}"

    return str_query, remaining_steps, initial_output


def generate_data_for_phrases(tokenizer, model, args):
    """
    Main function that:
      1. Reads English data
      2. Reads translated data for each language
      3. Builds an initial query (base_query_generate)
      4. Iteratively queries the model step-by-step, building final results
      5. Saves intermediate progress
    """
    # Build dictionary from English data
    phrases = ["train"]
    en_id2sample = {}
    for phrase in phrases:
        filename = f'data/en/{phrase}.json'
        data = read_jsonl_file(filename)
        for item in data:
            en_id2sample[item["id"]] = item

    # We will cycle through these languages
    root_langs = ["de", "es", "fr", "hi", "th"]
    final_result = {lang: {} for lang in root_langs}

    # This queue tracks items that still have steps to be generated
    # Keys = input length to help group queries of similar size
    llm_queue = {}
    queue_length = 0

    # Fill the queue with initial queries
    translated_path = "google_" if args.google_translated else ""

    for lang in root_langs:
        for phrase in phrases:
            filename = f'data/en/{translated_path}translated_{lang}_{phrase}.json'
            data = read_jsonl_file(filename)

            for item in tqdm(data, desc=f"Preparing {lang}-{phrase} items"):
                str_query, remaining_step_labels, current_llm_output = base_query_generate(
                    translated_item=item,
                    en_item=en_id2sample[item["id"]],
                    lang=lang
                )
                final_result[lang][item["id"]] = {
                    "raw_utterance": item["utterance"],
                    "translated_utterance": item["translated_utterance"],
                    "raw_decoupled_form": en_id2sample[item["id"]]["decoupled_form"],
                    "remain_step_labels": remaining_step_labels,
                    "num_remain_step_labels": len(remaining_step_labels),
                    "current_llm_query": str_query,
                    "current_llm_output": [current_llm_output],
                    "processed_step": 1
                }

                # Determine the input sequence length for grouping
                len_seq = tokenizer(
                    final_result[lang][item["id"]]["current_llm_query"],
                    return_tensors="pt"
                )["input_ids"].shape[-1]

                # If there are still steps left to generate, push to the queue
                if len(remaining_step_labels[1:]) != 0:
                    llm_queue.setdefault(len_seq, []).append(
                        {"id": item["id"], "lang": lang}
                    )
                    queue_length += 1

            # Save partial results right away
            out_filename = f'data/en/symmetry_step2step__{translated_path}llm_processed_{lang}_{phrase}.json'
            write_json(final_result[lang], out_filename)

    # Now iteratively consume the queue
    cur_step = 0
    while queue_length > 0:
        # Randomly pick a sequence length bucket to process
        random_sample_len = random.choice(list(llm_queue.keys()))
        batch_items = llm_queue[random_sample_len][: args.batch_size]
        llm_queue[random_sample_len] = llm_queue[random_sample_len][args.batch_size:]
        if not llm_queue[random_sample_len]:
            del llm_queue[random_sample_len]

        queries, ids, langs = [], [], []
        for item in batch_items:
            qid = item["id"]
            lang = item["lang"]
            ids.append(qid)
            langs.append(lang)
            queries.append(final_result[lang][qid]["current_llm_query"])
            queue_length -= 1

        # Generate batch output
        q_outputs = query_model(
            queries,
            tokenizer,
            model,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0
        )

        # Process each result
        for qid, lang, q_output in zip(ids, langs, q_outputs):
            # Extract the new step output from the model's text
            new_step_output = extract_current_step(
                output_text=q_output,
                num_processed_steps=len(final_result[lang][qid]["current_llm_output"])
            )

            # Append new step to the chain-of-thought
            final_result[lang][qid]["current_llm_output"].append(new_step_output)
            final_result[lang][qid]["remain_step_labels"] = final_result[lang][qid]["remain_step_labels"][1:]
            final_result[lang][qid]["num_remain_step_labels"] -= 1
            final_result[lang][qid]["processed_step"] += 1

            # Build an updated query for the next iteration
            final_result[lang][qid]["current_llm_query"] = build_new_query(
                current_query=final_result[lang][qid]["current_llm_query"],
                remaining_step_labels=final_result[lang][qid]["remain_step_labels"],
                new_output_step=new_step_output,
                current_llm_outputs=final_result[lang][qid]["current_llm_output"]
            )

            # If there are more steps to generate, re-insert into queue
            if final_result[lang][qid]["num_remain_step_labels"] > 0:
                updated_len_seq = tokenizer(
                    final_result[lang][qid]["current_llm_query"],
                    return_tensors="pt"
                )["input_ids"].shape[-1]

                llm_queue.setdefault(updated_len_seq, []).append(
                    {"id": qid, "lang": lang}
                )
                queue_length += 1

        # Periodically save progress
        if cur_step % args.save_step == 0:
            print(f"Saving at step {cur_step} - Remaining {queue_length} ...")
            for lang in root_langs:
                out_filename = f'data/en/symmetry_step2step__{translated_path}llm_processed_{lang}_train.json'
                write_json(final_result[lang], out_filename)
            print("Save DONE!")
        cur_step += 1

    print("All items processed. Final saving...")
    # Final save
    for lang in root_langs:
        out_filename = f'data/en/symmetry_step2step__{translated_path}llm_processed_{lang}_train.json'
        write_json(final_result[lang], out_filename)
    print("Completed generation for all languages.")

################################################################################
# Main Entry
################################################################################

def main():
    """
    Main entry point. Parses arguments, loads model, and processes data.
    """
    args = parse_arguments()
    print("Arguments:", args)

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Load model & tokenizer
    print("Loading model & tokenizer...")
    tokenizer, model = load_model_and_tokenizer(args.model)
    
    # Generate step-by-step queries
    generate_data_for_phrases(tokenizer, model, args)


if __name__ == "__main__":
    main()
