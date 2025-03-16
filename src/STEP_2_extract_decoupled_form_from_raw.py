import csv
import json
import re
import typing as t
from tqdm import tqdm

# External / Local Utilities
from extracting_utils import restore_full_logical_form, read_json_file, thai_tokenizer_2
from tree import Tree


###############################################################################
# Byte/Codepoint Constants & Functions
###############################################################################
TWOBYTES = 0x80
THREEBYTES = 0x800
FOURBYTES = 0x10000


def map_byte_to_codepoint_offset(text: str) -> t.List[int]:
    """
    Maps each UTF-8 byte offset to its corresponding codepoint offset.
    For multi-byte characters, duplicate offsets are inserted.

    :param text: The string for which to create the mapping
    :return: A list of codepoint offsets, with duplicates for multi-byte chars
    """
    mapping = []
    for codepoint_offset, character in enumerate(text):
        mapping.append(codepoint_offset)
        codepoint = ord(character)
        for boundary in (TWOBYTES, THREEBYTES, FOURBYTES):
            if codepoint >= boundary:
                mapping.append(codepoint_offset)
            else:
                break
    return mapping


###############################################################################
# TSV/JSON Reading and Writing
###############################################################################
def read_jsonl_file(file_path: str) -> t.List[t.Dict]:
    """
    Read a JSONL file and return its data as a list of dictionaries.

    :param file_path: Path to the JSONL file
    :return: List of dictionary records
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def read_tsv_file(filename: str) -> t.List[t.Dict]:
    """
    Read a TSV file and return its data as a list of dictionaries.

    :param filename: Path to the TSV file
    :return: List of dicts with keys from a predefined header
    """
    data = []
    with open(filename, 'r', newline='', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        headers = ["id", "intent", "slot_string", "utterance",
                   "domain", "locale", "decoupled_form", "tokens_json"]
        for row in reader:
            record = {
                headers[i]: value for i, value in enumerate(row)
            }
            data.append(record)
    return data


def write_tsv_file(data: t.List[t.Dict], file_path: str) -> None:
    """
    Write a list of dictionaries to a TSV file with specific columns.

    :param data: List of dicts that must include keys: 
                 ["id", "locale", "utterance", "full_logical_form"]
    :param file_path: Path where TSV should be written
    """
    ids = [x["id"] for x in data]
    locales = [x["locale"] for x in data]
    utterances = [x["utterance"] for x in data]
    full_logical_forms = [x["full_logical_form"] for x in data]

    with open(file_path, 'w', newline='', encoding="utf-8") as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for row_id, locale, utt, full_form in zip(ids, locales, utterances, full_logical_forms):
            # Additional columns can be inserted here if needed
            tsv_output.writerow([row_id, locale, utt, utt, full_form])


###############################################################################
# Miscellaneous String and Token Utilities
###############################################################################
def split_keep_delimiter(text: str, delimiters: t.List[str]) -> str:
    """
    Splits a string by multiple delimiters, but then re-inserts the delimiters 
    between tokens (with spaces). This helps when we want to keep the delimiters in text.

    :param text: The input string
    :param delimiters: List of delimiters to split on
    :return: A string with delimiters re-inserted, surrounded by spaces
    """
    new_text = text
    for delimiter in delimiters:
        parts = new_text.split(delimiter)
        new_text = f" {delimiter} ".join(parts)
    return new_text


def universal_tokenizer(utterance: str) -> str:
    """
    Example universal tokenizer that currently routes to a Thai tokenizer if needed.

    :param utterance: Input text string
    :return: Tokenized string
    """
    return thai_tokenizer_2(utterance)


###############################################################################
# Decoupled Form Reordering and Out-of-Slot Spans
###############################################################################
def reorder_decoupled_form(item: t.Dict) -> t.Dict:
    """
    Reorders the slots in the decoupled_form based on the slot_string field.

    :param item: Dictionary with keys ["slot_string", "decoupled_form", "intent"]
    :return: The same dictionary with 'decoupled_form' reordered in place
    """
    if item["slot_string"] == "":
        return item

    # Extract slot names from the slot_string
    list_slots = [slot.split(":")[-1] for slot in item["slot_string"].split(",")]
    list_slots = [slot for slot in list_slots if slot]

    # Extract the slots from the decoupled_form
    count_bracket = 0
    tokens = item["decoupled_form"].split(" ")
    decoupled_slots = []
    current_slot = []

    for token in tokens:
        if count_bracket != 0:
            current_slot.append(token)

        if count_bracket == 2 and token.startswith("]"):
            # current_slot[0] is something like "[slot_name"
            decoupled_slots.append((current_slot[0].split(":")[-1], " ".join(current_slot)))
            current_slot = []

        if token.startswith("["):
            count_bracket += 1
        elif token.startswith("]"):
            count_bracket -= 1

    # Align the order of decoupled slots with the order in slot_string
    correct_order_slots = []
    for slot_name in list_slots:
        for idx, dec_slot in enumerate(decoupled_slots):
            if slot_name == dec_slot[0]:
                # dec_slot is (slot_name, slot_string_with_brackets)
                correct_order_slots.append(decoupled_slots.pop(idx)[1])
                break

    # Build a new decoupled_form
    item["decoupled_form"] = f"[{item['intent']} {' '.join(correct_order_slots)} ]"
    return item


def get_slot_by_position(start: int, end: int, token_dict: t.Dict) -> str:
    """
    Builds a substring from token_dict's tokens whose spans fall between
    the 'start' and 'end' character positions.

    :param start: Start character index
    :param end: End character index
    :param token_dict: A dict with "tokens" and "tokenSpans" 
                       Each tokenSpan is a dict with "start" index
    :return: Space-joined substring derived from the tokens in range
    """
    result = []
    for token, token_span in zip(token_dict["tokens"], token_dict["tokenSpans"]):
        if start < token_span["start"] < end:
            result.append(token)
    return " ".join(result)


def extract_outslot_spans(item: t.Dict) -> t.List[str]:
    """
    Extract segments of the utterance that are not part of any slot.

    :param item: A dict with keys ["slot_string", "utterance", "locale"]
    :return: List of strings that lie outside the defined slot spans
    """
    slot_strs = [slot for slot in item["slot_string"].split(",") if slot]
    char_offset_map = map_byte_to_codepoint_offset(item["utterance"])

    outslot_spans = []
    start_index = char_offset_map[0] if char_offset_map else 0

    for slot in slot_strs:
        start, end = slot.split(":")
        end_index = int(start)
        # substring from current start_index up to the beginning of the slot
        span_text = item["utterance"][start_index:char_offset_map[end_index]].strip()
        if "th" in item["locale"]:
            span_text = thai_tokenizer_2(span_text)
        outslot_spans.append(span_text)

        # Move start_index to slot's end
        slot_end_index = int(end)
        if slot_end_index < len(char_offset_map):
            start_index = char_offset_map[slot_end_index]
        else:
            start_index = slot_end_index

    # Add trailing substring after the last slot
    tail_text = item["utterance"][start_index:].strip()
    outslot_spans.append(thai_tokenizer_2(tail_text) if "th" in item["locale"] else tail_text)

    return outslot_spans


def extract_outslot_spans_v2(item: t.Dict) -> t.List[str]:
    """
    Alternate version to extract out-of-slot spans. Uses token-based positions 
    from "tokens_json" instead of raw character offsets.

    :param item: Dict with keys ["slot_string", "utterance", "tokens_json"]
    :return: List of outslot spans
    """
    slot_strs = [slot for slot in item["slot_string"].split(",") if slot]
    token_dict = json.loads(item["tokens_json"])

    outslot_spans = []
    start = -1
    for slot in slot_strs:
        slot_start = int(slot.split(":")[0])
        span_text = get_slot_by_position(start, slot_start, token_dict)
        outslot_spans.append(span_text)
        start = int(slot.split(":")[1])

    # Capture trailing substring
    last_token_start = token_dict["tokenSpans"][-1]["start"] + 1
    tail_text = item["utterance"][start:].strip()
    outslot_spans.append(tail_text)
    return outslot_spans


###############################################################################
# Full Logical Form Builders
###############################################################################
def build_full_logical_form(item: t.Dict) -> str:
    """
    Constructs the full logical form by inserting out-of-slot spans into a
    decoupled form. 
    Uses extract_outslot_spans for out-of-slot text.

    :param item: Dict that must contain keys 
                 ["slot_string", "decoupled_form", "tokens_json", "intent", "utterance"]
    :return: A string representing the full logical form
    """
    # Reorder decoupled_form to match slot_string's order
    item = reorder_decoupled_form(item)

    # If no slots, just wrap the entire utterance
    if not item["slot_string"].strip():
        tokens = json.loads(item["tokens_json"])["tokens"]
        utterance = " ".join(tokens)
        return f"[{item['intent']} {utterance} ]"

    # Insert outslot spans where needed
    count_bracket = 0
    current_outslot_span = 0
    outslot_spans = extract_outslot_spans(item)
    tokens = item["decoupled_form"].split()
    result = []

    for token in tokens:
        # When count_bracket == 1, inserting a bracket or closing bracket 
        # indicates a new out-of-slot region
        if (count_bracket == 1 and token.startswith("[")) or (count_bracket == 1 and token.startswith("]")):
            result.append(outslot_spans[current_outslot_span])
            current_outslot_span += 1

        result.append(token)
        if token.startswith("["):
            count_bracket += 1
        elif token.startswith("]"):
            count_bracket -= 1

    # Normalize spaces
    return re.sub(r'\s+', ' ', " ".join(result)).strip()


def build_full_logical_form_v2(item: t.Dict) -> str:
    """
    Similar to build_full_logical_form, but uses extract_outslot_spans_v2 
    for out-of-slot text retrieval.

    :param item: Dict that must contain keys 
                 ["slot_string", "decoupled_form", "tokens_json", "intent", "utterance"]
    :return: A string representing the full logical form
    """
    item = reorder_decoupled_form(item)

    if not item["slot_string"].strip():
        tokens = json.loads(item["tokens_json"])["tokens"]
        utterance = " ".join(tokens)
        return f"[{item['intent']} {utterance} ]"

    count_bracket = 0
    current_outslot_span = 0
    outslot_spans = extract_outslot_spans_v2(item)
    tokens = item["decoupled_form"].split()
    result = []

    for token in tokens:
        # If we hit a bracket at level 1, insert out-of-slot text
        if (count_bracket == 1 and token.startswith("[")) or (count_bracket == 1 and token.startswith("]")):
            result.append(outslot_spans[current_outslot_span])
            current_outslot_span += 1

        result.append(token)
        if token.startswith("["):
            count_bracket += 1
        elif token.startswith("]"):
            count_bracket -= 1

    return re.sub(r'\s+', ' ', " ".join(result)).strip()


###############################################################################
# Schema Extraction
###############################################################################
def extract_schema(decoupled_form: str) -> t.Tuple[t.List[str], t.List[str]]:
    """
    Split the decoupled_form tokens into bracket tokens and span tokens.

    :param decoupled_form: String like "[INTENT [SLOT some span ] ... ]"
    :return: (list_of_brackets, list_of_span_words)
    """
    tokens = decoupled_form.split()
    logical_tokens = []
    span_tokens = []

    for token in tokens:
        if token.startswith("[") or token.startswith("]"):
            logical_tokens.append(token)
        else:
            span_tokens.append(token)

    return logical_tokens, span_tokens


###############################################################################
# Building Various Data Sets
###############################################################################
def build_eval_data(folder_path: str, phase: str) -> None:
    """
    Create data for MRC-style evaluation from TSV files, storing context 
    and original decoupled form.

    :param folder_path: Directory containing subfolders by language
    :param phase: Either "eval" or "test"
    """
    print(f" =========== Processing {phase} data ... =========== ")
    langs = ["de", "es", "fr", "hi", "th"]
    data = []
    for lang in langs:
        raw_data = read_tsv_file(f'{folder_path}/{lang}/{phase}.txt')
        for item in tqdm(raw_data, desc=f"{lang}-{phase}"):
            # Convert tokens_json -> utterance
            utterance = " ".join(json.loads(item["tokens_json"])["tokens"])
            data.append({
                "qas_id": item["id"],
                "context": utterance,
                "org_label": item["decoupled_form"],
                "locale": item["locale"].split("_")[0]
            })

    out_file = f"data/mrc-ner.{phase}"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_train_en_only_data(folder_path: str) -> None:
    """
    Builds a training set for English only, extracting a "full_logical_form" 
    for each record and writing the result to a TSV.

    :param folder_path: Directory with "en/train.txt"
    """
    print(f" =========== Processing train data (English only) ... =========== ")
    raw_data = read_tsv_file(f'{folder_path}/en/train.txt')
    data = []
    for item in tqdm(raw_data, desc="en-train"):
        full_logical_form = build_full_logical_form(item).replace('"', "'")
        _, span_tokens = extract_schema(full_logical_form)
        utterance = " ".join(span_tokens)
        data.append({
            "id": item["id"],
            "locale": item["locale"].split("_")[0],
            "utterance": utterance,
            "full_logical_form": full_logical_form
        })

    write_tsv_file(data, "data/raw/train_en_only.tsv")


def build_train_oracle_data(folder_path: str) -> None:
    """
    Creates a training set for multiple languages ("en", "de", "es", "fr", "hi", "th")
    using the "oracle" approach, and writes it to a TSV.

    :param folder_path: Base directory containing each lang subfolder with 'train.txt'
    """
    print(f" =========== Processing train data (oracle, multiple langs) ... =========== ")
    data = []
    langs = ["en", "de", "es", "fr", "hi", "th"]
    for lang in langs:
        raw_data = read_tsv_file(f'{folder_path}/{lang}/train.txt')
        for item in tqdm(raw_data, desc=f"{lang}-train"):
            full_logical_form = build_full_logical_form(item).replace('"', "'").replace("​", "")
            _, span_tokens = extract_schema(full_logical_form)
            utterance = " ".join(span_tokens)
            data.append({
                "id": item["id"],
                "locale": item["locale"].split("_")[0],
                "utterance": utterance,
                "full_logical_form": full_logical_form
            })

    write_tsv_file(data, "data/raw/train_oracle.tsv")


def iserror(func, *args, **kw) -> bool:
    """
    Helper to check if calling a function raises an Exception.

    :return: True if an exception is raised, otherwise False.
    """
    try:
        func(*args, **kw)
        return False
    except Exception:
        return True

def format_output_cot(output_text, num_candidates=1, num_steps=100):
    cleaned_output_text = output_text.replace("<s>", "").replace("<pad> ", "").replace("</s>", "")
    output_sample = cleaned_output_text.split("\n\n")[num_candidates]
    predicted_steps = output_sample.split("\n")[:num_steps+1]
    result = []
    pattern = r"Step \d+: "  # Regex pattern to match "Step [number]: "
    for step in predicted_steps:
        if step.startswith("Step"):
            tmp = re.sub(pattern, "", step).strip()
            result.append(tmp)
    return result



def build_train_augmented_data(folder_path: str) -> None:
    """
    Build an augmented training set from multiple sources (symmetry, mBART, gold, continuous, standard).
    Reads previously saved JSON, merges them with original English data, 
    tries to restore full logical forms, and writes to TSV.

    :param folder_path: Directory with subfolders and JSON expansions
    """
    # 1. Collect original English data
    raw_data = read_tsv_file(f'{folder_path}/en/train.txt')
    en_data = []
    for item in tqdm(raw_data, desc="Collect en train"):
        full_form = build_full_logical_form(item).replace('"', "'")
        _, span_tokens = extract_schema(full_form)
        utterance = " ".join(span_tokens)
        en_data.append({
            "id": item["id"],
            "locale": item["locale"].split("_")[0],
            "utterance": utterance,
            "full_logical_form": full_form
        })

    # 2. Build from "symmetry" expansions
    langs = ["de", "es", "fr", "hi", "th"]
    # for size in ["", "llama2_7b_", "llama2_13b_"]:  # Potential expansions
    for size in [""]:  # If only using the base size
        result_out = []
        for lang in langs:
            print(f"====== {lang} (symmetry, size={size}) ======")
            filename = f'{folder_path}/en/{size}symmetry_step2step__google_llm_processed_{lang}_train.json'
            raw_sym_data = read_json_file(filename)
            total, error_count = 0, 0
            for rec_id, rec_val in raw_sym_data.items():
                total += 1
                try:
                    full_logical_form = restore_full_logical_form(
                        rec_val["current_llm_output"],
                        th_tokenizer="thai_tokenizer_2",
                        lang=lang
                    ).replace('"', "'")
                    # Validate parse
                    Tree(full_logical_form)

                    _, span_tokens = extract_schema(full_logical_form)
                    utterance = " ".join(span_tokens)
                    result_out.append({
                        "id": rec_id,
                        "locale": lang,
                        "utterance": utterance,
                        "full_logical_form": full_logical_form
                    })
                except:
                    error_count += 1
            print("Error", error_count, "Total", total)
        # Merge English with newly created expansions
        outfile = f"data/raw/train_with_augumented_{size}data.tsv"
        write_tsv_file(en_data + result_out, outfile)

    # 3. mBART and gold translation expansions
    for setting in ["mbart", "gold"]:
        result_out = []
        for lang in langs:
            print(f"====== {lang} (setting={setting}) ======")
            filename = f'{folder_path}/en/symmetry_step2step__{setting}__llm_processed_{lang}_train.json'
            raw_trans_data = read_json_file(filename)

            total, error_count = 0, 0
            for rec_id, rec_val in raw_trans_data.items():
                total += 1
                try:
                    full_logical_form = restore_full_logical_form(
                        rec_val["current_llm_output"],
                        th_tokenizer="thai_tokenizer_2",
                        lang=lang
                    ).replace('"', "'").replace("​", "")
                    # Validate parse
                    Tree(full_logical_form)

                    _, span_tokens = extract_schema(full_logical_form)
                    utterance = " ".join(span_tokens)
                    result_out.append({
                        "id": rec_id,
                        "locale": lang,
                        "utterance": utterance,
                        "full_logical_form": full_logical_form
                    })
                except:
                    error_count += 1
            print("Error", error_count, "Total", total)

        outfile = f"data/raw/train_with_augumented_data_{setting}_translation.tsv"
        write_tsv_file(en_data + result_out, outfile)

    # 4. "continuous" expansions
    result_out = []
    for lang in langs:
        print(f"====== {lang} (continuous) ======")
        filename = f'{folder_path}/en/continuous_step2step__google_llm_processed_{lang}_train.json'
        raw_cont_data = read_json_file(filename)

        total, error_count = 0, 0
        for rec_id, rec_val in raw_cont_data.items():
            total += 1
            try:
                full_logical_form = restore_full_logical_form(
                    rec_val["current_llm_output"],
                    th_tokenizer="thai_tokenizer_2",
                    lang=lang
                ).replace('"', "'")
                Tree(full_logical_form)

                _, span_tokens = extract_schema(full_logical_form)
                utterance = " ".join(span_tokens)
                result_out.append({
                    "id": rec_id,
                    "locale": lang,
                    "utterance": utterance,
                    "full_logical_form": full_logical_form
                })
            except:
                error_count += 1
        print("Error", error_count, "Total", total)

    outfile = "data/raw/train_with_augumented_continuous_data.tsv"
    write_tsv_file(en_data + result_out, outfile)

    # 5. Standard CoT expansions
    id2sample = {}
    en_train_jsonl = "../data/en/train.json"
    en_train_data = read_jsonl_file(en_train_jsonl)
    for entry in tqdm(en_train_data, desc="Index en/train.json"):
        id2sample[entry["id"]] = entry

    result_out = []
    for lang in langs:
        print(f"====== {lang} (standard COT) ======")
        filename = f"data/en/google_llm_processed_{lang}_train.json"
        raw_std_cot = read_jsonl_file(filename)

        total, error_count = 0, 0
        for rec_item in tqdm(raw_std_cot, desc=f"{lang}-stdCOT"):
            total += 1
            record_id = rec_item["id"]
            support_sample = id2sample.get(record_id)
            if not support_sample:
                error_count += 1
                continue

            num_steps = support_sample["decoupled_form"].count("]")
            try:
                # Convert raw_llm_decoupled_form to chain-of-thought lines
                cot_lines = format_output_cot(
                    rec_item["raw_llm_decoupled_form"],
                    num_candidates=1,
                    num_steps=num_steps
                )
                full_logical_form = restore_full_logical_form(
                    cot_lines,
                    th_tokenizer="thai_tokenizer_2",
                    lang=lang
                ).replace('"', "'")

                _, span_tokens = extract_schema(full_logical_form)
                utterance = " ".join(span_tokens)
                result_out.append({
                    "utterance": utterance,
                    "full_logical_form": full_logical_form,
                    "id": record_id,
                    "locale": lang
                })
            except:
                error_count += 1

        print("Error", error_count, "Total", total)

    outfile = "data/augmented/train_with_augumented_standard_data.tsv"
    write_tsv_file(en_data + result_out, outfile)


###############################################################################
# Main Build Pipeline
###############################################################################
def build_data():
    """
    Main pipeline to build all data sets:
      1. Eval data (eval and test)
      2. train_en_only_data
      3. train_oracle_data
      4. train_augmented_data
    """
    folder_path = "data"

    # Build test & eval data
    build_eval_data(folder_path, "eval")
    build_eval_data(folder_path, "test")

    # Build English-only train data
    build_train_en_only_data(folder_path)

    # Build multi-language oracle train data
    build_train_oracle_data(folder_path)

    # Build augmented data with expansions
    build_train_augmented_data(folder_path)


if __name__ == "__main__":
    build_data()
