import json
import re
from tree import Tree, Root, Token

# External libraries for Thai text segmentation
from pythainlp.tokenize import word_tokenize
from thai_tokenizer import Tokenizer


###############################################################################
# Constants
###############################################################################
OPEN_BRACKET = "["
CLOSE_BRACKET = "]"


###############################################################################
# I/O Utilities
###############################################################################
def save_to_jsonl(data_list, filename):
    """
    Save a list of Python dictionaries to a JSONL file.

    :param data_list: List of dict objects to be serialized
    :param filename: Output file path for the JSONL
    """
    with open(filename, 'w', encoding='utf-8') as jsonl_file:
        for item in data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            jsonl_file.write(json_line + '\n')


def read_jsonl_file(file_path):
    """
    Read a JSONL file and return a list of Python dictionaries.

    :param file_path: Path to the JSONL file
    :return: List of dict objects
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def write_json(data, path):
    """
    Write Python data to a JSON file with indentation.

    :param data: Python object (likely dict or list) to write
    :param path: Output file path
    """
    with open(path, 'w', encoding='utf-8') as f:
        text_data = json.dumps(data, indent=4, ensure_ascii=False)
        f.write(text_data)


def read_json_file(path):
    """
    Read a JSON file and return its contents as a Python object.

    :param path: Path to the JSON file
    :return: Python object (dict or list) from the JSON
    """
    with open(path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)


###############################################################################
# Thai Tokenizers
###############################################################################
thai_tokenizer_1 = Tokenizer()

def thai_tokenizer_2(text):
    """
    Alternate Thai tokenizer that preserves bracket format if present.

    :param text: Input string
    :return: Tokenized string
    """
    if text == "":
        return ""

    tokens = text.split()
    # If text is bracketed [ ... ], tokenize only what's inside
    if tokens[0].startswith(OPEN_BRACKET) and tokens[-1].startswith(CLOSE_BRACKET):
        processed_text = word_tokenize(" ".join(tokens[1:-1]), engine="icu")
        merged_text = re.sub(' +', ' ', " ".join(processed_text)).strip()
        tokens = [tokens[0]] + [merged_text] + [tokens[-1]]
    else:
        processed_text = word_tokenize(text, engine="icu")
        merged_text = re.sub(' +', ' ', " ".join(processed_text)).strip()
        tokens = [merged_text]
    return " ".join(tokens)


###############################################################################
# Span & Logical Form Utilities
###############################################################################
def find_sub_list(short_list, long_list):
    """
    Find all start/end indices of short_list's occurrences in long_list (case-insensitive).

    :param short_list: List of tokens (shorter sequence)
    :param long_list: List of tokens (longer sequence)
    :return: List of (start_index, end_index) tuples
    """
    sl = [item.lower() for item in short_list]
    ll = [item.lower() for item in long_list]

    results = []
    sll = len(sl)
    for idx in (i for i, e in enumerate(ll) if e == sl[0]):
        if ll[idx : idx + sll] == sl:
            results.append((idx, idx + sll - 1))
    return results


def remove_end_punctuation(cot_steps):
    """
    Remove trailing punctuation ('?' or '.') in the last token before a closing bracket.

    :param cot_steps: List of strings representing chain-of-thought steps
    :return: Updated list of CoT steps
    """
    updated_steps = []
    for step in cot_steps:
        tokens = step.split()
        index = len(tokens) - 1

        # Move index back if we have closing brackets
        while tokens[index].startswith(CLOSE_BRACKET):
            index -= 1

        # If last token ends with '?' or '.', separate it out
        last_char = tokens[index][-1]
        if last_char in ("?", "."):
            tokens[index] = tokens[index][:-1]  # Remove punctuation
            tokens.insert(index + 1, last_char) # Add punctuation as a separate token

        updated_steps.append(" ".join(tokens))
    return updated_steps


###############################################################################
# Tree Parsing & Node Extraction
###############################################################################
def list_nonterminals(cur_node, node_info, node_level):
    """
    Recursively collect non-terminal children from a given node in the parse tree.

    :param cur_node: Current node of the parse tree
    :param node_info: Accumulating list of (span_str, node_level, label)
    :param node_level: Current level in the tree
    :return: Updated list of node_info
    """
    for child in cur_node.children:
        # Only consider non-terminal children
        if not isinstance(child, (Root, Token)):
            span = get_span(child)
            span_str = "0;0"
            if span is not None:
                # The node's span typically includes the end index as exclusive, so subtract 1
                span_str = f"{span[0]};{span[1] - 1}"

            node_info.append((span_str, node_level, child.label))

    # Recurse deeper
    for child in cur_node.children:
        if not isinstance(child, (Root, Token)):
            list_nonterminals(child, node_info, node_level + 1)
    return node_info


def get_span(node):
    """
    Return the start and end token indices of the given node.

    :param node: Parse tree node
    :return: Tuple (start_idx, end_idx)
    """
    return node.get_token_span()


def get_node_info(tree):
    """
    Retrieve label, level, and span info for all non-terminal nodes in a tree.

    :param tree: An instance of Tree
    :return: List of tuples (span_str, node_level, node_label)
    """
    node_info = list_nonterminals(tree.root, [], 1)
    return node_info


###############################################################################
# Building Chain-of-Thought / Labels
###############################################################################
def build_cot(input_text, label_types, entity_levels, start_positions, end_positions):
    """
    Build a list of bracketed label strings from the decoupled form.

    :param input_text: Original text/utterance
    :param label_types: List of label names (e.g., INTENT, SLOT, etc.)
    :param entity_levels: List of entity levels (unused in final string, but useful for ordering)
    :param start_positions: List of start indices for each label
    :param end_positions: List of end indices for each label
    :return: List of bracketed label strings
    """
    # Remove existing bracket tokens to avoid double brackets
    tokens = [
        token for token in input_text.split()
        if not (token.startswith(OPEN_BRACKET) or token.startswith(CLOSE_BRACKET))
    ]
    result_list = []

    for idx, (label_type, ent_level, start_pos, end_pos) in enumerate(
        zip(label_types, entity_levels, start_positions, end_positions)
    ):
        if idx == 0:
            # For the first label, we don't include the tokens in the bracket
            result_list.append(f"[{label_type} ]")
        else:
            snippet = " ".join(tokens[start_pos : end_pos + 1])
            result_list.append(f"[{label_type} {snippet} ]")

    return result_list


def extract_label_list(decoupled_form):
    """
    Given a decoupled form string, parse it into bracket-labeled components.

    :param decoupled_form: String with bracket labels, e.g. [INTENT ask_weather ]
    :return: A list of bracketed label strings (the "chain of thought")
    """
    # Build a tree from the decoupled form
    node_info = get_node_info(Tree(decoupled_form))

    # Extract label info
    label_types = [info[2] for info in node_info]
    entity_levels = [info[1] for info in node_info]
    start_positions = [int(info[0].split(";")[0]) for info in node_info]
    end_positions = [int(info[0].split(";")[1]) for info in node_info]

    # Build bracketed CoT
    return build_cot(decoupled_form, label_types, entity_levels, start_positions, end_positions)


###############################################################################
# Restoring Full Logical Forms
###############################################################################
def restore_full_logical_form(
    cot_steps,
    lang="en",
    tokenizer_func=None,
    delete_end_punctuation=True
):
    """
    Re-insert bracket labels into a token sequence to reconstruct the full logical form.
    Includes an option to use a Thai tokenizer for token alignment.

    :param cot_steps: List of bracketed label strings (chain-of-thought steps)
    :param lang: Language code (e.g., 'en', 'th', etc.)
    :param tokenizer_func: Callable to tokenize a given string (used for Thai)
    :param delete_end_punctuation: Whether to strip punctuation before bracket insertion
    :return: String representing the reconstructed logical form
    """
    if not cot_steps:
        return ""

    if delete_end_punctuation:
        cot_steps = remove_end_punctuation(cot_steps)

    # Tokenize the first step if needed
    if lang == "th" and tokenizer_func:
        root_tokens = tokenizer_func(cot_steps[0]).split()
    else:
        root_tokens = cot_steps[0].split()

    # Process remaining steps
    for step in cot_steps[1:]:
        if lang == "th" and tokenizer_func:
            cur_tokens = tokenizer_func(step).split()
        else:
            cur_tokens = step.split()

        label_token = cur_tokens[0]
        close_bracket_token = cur_tokens[-1]
        span_tokens = cur_tokens[1:-1]

        # Validate bracket format
        if not (label_token.startswith(OPEN_BRACKET) and close_bracket_token == CLOSE_BRACKET):
            raise ValueError(f'"{step}" is invalid sub-logic (missing bracket).')

        # Find positions of the span in the main token list
        span_indices = find_sub_list(span_tokens, root_tokens)
        if not span_indices:
            raise ValueError(f'Cannot find span "{step}" in the utterance {root_tokens}.')

        # Insert bracket tokens around each found span (reverse order to preserve indices)
        for start_idx, end_idx in reversed(span_indices):
            root_tokens.insert(end_idx + 1, close_bracket_token)
            root_tokens.insert(start_idx, label_token)

    return " ".join(root_tokens)


def extract_full_form(decoupled_form, original_tokens, delete_end_punctuation=False):
    """
    Merge an existing decoupled form with the raw tokens of an utterance,
    reconstructing the bracketed logical form.

    :param decoupled_form: Original decoupled form string
    :param original_tokens: List of tokens from the original utterance
    :param delete_end_punctuation: Whether to remove punctuation before bracket insertion
    :return: String representing the reconstructed logical form
    """
    # Extract bracket steps from the decoupled form
    steps = extract_label_list(decoupled_form)

    # The first step bracket is usually the intent with no tokens, so
    # we add the original utterance tokens to it
    if steps:
        first_step = steps[0]
        # Insert the original tokens into the first bracket
        # Example: [INTENT ] => [INTENT do you know the weather ]
        if first_step.endswith(" ]"):
            steps[0] = first_step[:-2] + " ".join(original_tokens) + " ]"

    # Use the simpler version of restore_full_logical_form without Thai logic
    return restore_full_logical_form(
        cot_steps=steps,
        lang="en",
        tokenizer_func=None,
        delete_end_punctuation=delete_end_punctuation
    )
