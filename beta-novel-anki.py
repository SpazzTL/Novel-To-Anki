import csv
import os
import pickle
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from typing import Any, Dict, List, Tuple

import ebooklib
from ebooklib import epub
from tqdm import tqdm

try:
    import orjson
except ImportError:
    import json as orjson

# --- Constants ---
CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
SENTENCE_SPLITTER = re.compile(r'(?<=[.?!。？！])\s+')
DICT_SOURCE_SPLITTER = re.compile(r'(<b>.*?<\/b>:)')

# --- i+1 and Sentence Configuration ---
# The maximum number of "unknown" vocabulary words allowed in an example sentence.
I_PLUS_ONE_THRESHOLD = 3
# The maximum number of example sentences to find for each word.
NUM_EXAMPLE_SENTENCES = 5
# The minimum number of meaningful words a sentence must have to be included.
MIN_SENTENCE_WORD_COUNT = 5


# Output Directories
DICTIONARIES_FOLDER = 'Dictionaries'
CACHE_FOLDER = 'cache'
ANKI_FOLDER = 'output/anki_csv'
GRAMMAR_FOLDER = 'output/grammar_analysis'
JSON_FOLDER = 'output/json_analysis'

# --- Definition Parsing ---
def parse_json_to_html(json_data: Any) -> str:
    """Recursively parses a custom JSON dictionary structure into a valid HTML string."""
    items = json_data if isinstance(json_data, list) else [json_data]
    html_parts = []

    for item in items:
        if isinstance(item, str):
            html_parts.append(item)
            continue
        if not isinstance(item, dict):
            continue

        tag = item.get('tag')
        content = item.get('content')
        style = item.get('style', {})
        lang = item.get('lang')

        style_attrs = ''
        if style:
            style_str_list = [
                f"{re.sub(r'(?<!^)(?=[A-Z])', '-', key).lower()}:{value};"
                for key, value in style.items()
            ]
            style_attrs = f' style="{"".join(style_str_list)}"'

        lang_attr = f' lang="{lang}"' if lang else ''
        attributes = f'{style_attrs}{lang_attr}'

        if tag:
            html_parts.append(f'<{tag}{attributes}>')
        if content:
            html_parts.append(parse_json_to_html(content))
        if tag:
            html_parts.append(f'</{tag}>')

    return ''.join(html_parts)

def parse_definition(definition_text: str) -> str:
    """Parses a raw definition string into clean, human-readable HTML."""
    if not isinstance(definition_text, str) or not definition_text.strip():
        return '<div class="definition-content">No definition found.</div>'

    parts = DICT_SOURCE_SPLITTER.split(definition_text)
    if len(parts) <= 1:
        try:
            if definition_text.strip().startswith(('[', '{')):
                data = orjson.loads(definition_text)
                return '<div class="definition-content">' + parse_json_to_html(data) + '</div>'
        except (orjson.JSONDecodeError, TypeError):
            pass
        return f'<div class="definition-content">{definition_text}</div>'

    cleaned_definitions = []
    for i in range(1, len(parts), 2):
        source_tag = parts[i]
        content_str = parts[i + 1].strip()
        processed_content = ""

        try:
            data = orjson.loads(content_str)
            if isinstance(data, list) and all(isinstance(item, str) for item in data):
                processed_content = '; '.join(data)
            else:
                processed_content = parse_json_to_html(data)
        except (orjson.JSONDecodeError, TypeError):
            processed_content = content_str

        if processed_content:
            cleaned_definitions.append(f"{source_tag} {processed_content}")

    return '<div class="definition-content">' + '<br>'.join(cleaned_definitions) + '</div>'

# --- File Processing & Text Extraction ---
def find_source_files() -> List[str]:
    """Scans common directories for .epub and .txt files."""
    files = []
    for directory in ['.', 'input', 'novels']:
        if os.path.isdir(directory):
            files.extend(glob(os.path.join(directory, '*.epub')))
            files.extend(glob(os.path.join(directory, '*.txt')))
    return files

def extract_text_from_epub(epub_path: str) -> str:
    """Extracts all text content from an EPUB file, cleaning HTML tags."""
    text_content = []
    try:
        book = epub.read_epub(epub_path)
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            try:
                raw_html = item.get_body_content().decode('utf-8', 'ignore')
                text_content.append(clean_html(raw_html))
            except Exception as e:
                print(f"WARNING: Skipping a malformed item in '{os.path.basename(epub_path)}' due to error: {e}")
        return "".join(text_content)
    except (zipfile.BadZipFile, KeyError) as e:
        print(f"\nWARNING: Skipping corrupt or malformed EPUB file: {os.path.basename(epub_path)} (Error: {e})")
        return ""
    except Exception as e:
        print(f"\nWARNING: An unexpected error occurred with file '{os.path.basename(epub_path)}': {e}")
        return ""

def extract_text_from_txt(txt_path: str) -> str:
    """Extracts text content from a .txt file."""
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def clean_html(raw_html: str) -> str:
    """Removes HTML tags and extra whitespace from a string."""
    cleantext = re.sub(CLEANR, '', raw_html)
    return re.sub(r'\s+', ' ', cleantext).strip()

def get_sentences(text: str, source_file: str) -> List[Dict]:
    """Splits a block of text into sentences, returning a list of dicts."""
    base_name = os.path.basename(source_file)
    return [
        {"sentence": s.strip(), "source": base_name}
        for s in SENTENCE_SPLITTER.split(text) if s.strip()
    ]

# --- Parallel Language Analysis ---
def initialize_korean_worker():
    """Initializes the KoNLPy Okt tagger for each worker process."""
    global okt
    from konlpy.tag import Okt
    okt = Okt()

def analyze_sentence_chunk_korean(sentence_chunk: List[Dict]) -> Tuple[Dict[str, int], List[str], List[Dict]]:
    """Analyzes a list of Korean sentences to extract words, grammar, and tokens."""
    words = Counter()
    grammar = []
    tokenized_sentences = []
    for sentence_data in sentence_chunk:
        sentence = sentence_data["sentence"]
        try:
            original_tokens = {token for token, pos in okt.pos(sentence, norm=True, stem=False)}
            pos_tagged_stemmed = okt.pos(sentence, norm=True, stem=True)

            meaningful_tokens = [word for word, pos in pos_tagged_stemmed if pos not in ['Punctuation', 'Foreign', 'Josa', 'Suffix']]
            if len(meaningful_tokens) < MIN_SENTENCE_WORD_COUNT:
                continue

            current_sentence_tokens = []
            for word, pos in pos_tagged_stemmed:
                is_valid_word = True
                current_sentence_tokens.append(word)

                if pos == 'Noun':
                    for original_token in original_tokens:
                        if word != original_token and word in original_token:
                            is_valid_word = False
                            break

                if is_valid_word and pos in ['Noun', 'Verb', 'Adjective', 'Adverb'] and len(word) > 1:
                    words[word] += 1
                
                if pos not in ['Punctuation', 'Foreign']:
                    grammar.append(f"{word}/{pos}")
            
            tokenized_sentences.append({
                "sentence": sentence,
                "tokens": tuple(current_sentence_tokens),
                "source": sentence_data["source"]
            })
        except Exception as e:
            print(f"\nWARNING: KoNLPy failed to process a sentence. Error: {e}")
            
    return dict(words), grammar, tokenized_sentences

def initialize_japanese_worker():
    """Initializes the Janome Tokenizer for each worker process."""
    global tokenizer
    from janome.tokenizer import Tokenizer
    tokenizer = Tokenizer()

def analyze_sentence_chunk_japanese(sentence_chunk: List[Dict]) -> Tuple[Dict[str, int], List[str], List[Dict]]:
    """Analyzes a list of Japanese sentences to extract words, grammar, and tokens."""
    words = Counter()
    grammar = []
    tokenized_sentences = []
    target_pos = ['名詞', '動詞', '形容詞', '副詞']
    ignore_pos_for_count = ['助詞', '助動詞', '記号', '接頭詞', '接続詞', '連体詞']

    for sentence_data in sentence_chunk:
        sentence = sentence_data["sentence"]
        try:
            tokens = list(tokenizer.tokenize(sentence))
            
            meaningful_tokens = [t for t in tokens if t.part_of_speech.split(',')[0] not in ignore_pos_for_count]
            if len(meaningful_tokens) < MIN_SENTENCE_WORD_COUNT:
                continue

            current_sentence_tokens = []
            for token in tokens:
                pos = token.part_of_speech.split(',')[0]
                word = token.base_form
                current_sentence_tokens.append(word)

                if pos in target_pos and len(word) > 1:
                    words[word] += 1
                if pos not in ['助詞', '助動詞', '記号']:
                    grammar.append(f"{token.surface}/{pos}")

            tokenized_sentences.append({
                "sentence": sentence,
                "tokens": tuple(current_sentence_tokens),
                "source": sentence_data["source"]
            })
        except Exception as e:
            print(f"\nWARNING: Janome failed to process a sentence. Error: {e}")
    return dict(words), grammar, tokenized_sentences

def initialize_spanish_worker():
    """
    Initializes the spaCy model for each worker process.
    Requires: pip install spacy && python -m spacy download es_core_news_sm
    """
    global nlp
    import spacy
    try:
        nlp = spacy.load("es_core_news_sm")
    except OSError:
        print("\nERROR: Spanish model not found. Please run 'python -m spacy download es_core_news_sm'")
        sys.exit(1)


def analyze_sentence_chunk_spanish(sentence_chunk: List[Dict]) -> Tuple[Dict[str, int], List[str], List[Dict]]:
    """Analyzes a list of Spanish sentences using spaCy."""
    words = Counter()
    grammar = []
    tokenized_sentences = []
    target_pos = ['NOUN', 'VERB', 'ADJ', 'ADV']

    for sentence_data in sentence_chunk:
        sentence = sentence_data["sentence"]
        try:
            doc = nlp(sentence)

            meaningful_tokens = [token for token in doc if not token.is_punct and not token.is_space]
            if len(meaningful_tokens) < MIN_SENTENCE_WORD_COUNT:
                continue
            
            current_sentence_tokens = []
            for token in doc:
                word = token.lemma_.lower()
                pos = token.pos_
                current_sentence_tokens.append(word)

                if pos in target_pos and len(word) > 1:
                    words[word] += 1
                
                if not token.is_punct and not token.is_space:
                    grammar.append(f"{token.text}/{pos}")
            
            tokenized_sentences.append({
                "sentence": sentence,
                "tokens": tuple(current_sentence_tokens),
                "source": sentence_data["source"]
            })

        except Exception as e:
            print(f"\nWARNING: spaCy failed to process a sentence. Error: {e}")
            
    return dict(words), grammar, tokenized_sentences


def analyze_text_parallel(text: str, language: str, source_file: str) -> Tuple[Dict[str, int], Counter, List[Dict]]:
    """Analyzes a large text in parallel, returning word counts, grammar, and tokenized sentences."""
    if language == 'korean':
        initializer = initialize_korean_worker
        analyzer = analyze_sentence_chunk_korean
    elif language == 'japanese':
        initializer = initialize_japanese_worker
        analyzer = analyze_sentence_chunk_japanese
    elif language == 'spanish':
        initializer = initialize_spanish_worker
        analyzer = analyze_sentence_chunk_spanish
    else:
        raise ValueError("Unsupported language specified.")

    sentences = get_sentences(text, source_file)
    word_counts = Counter()
    grammar = Counter()
    all_tokenized_sentences = []
    chunk_size = 2000
    sentence_chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]

    with ProcessPoolExecutor(initializer=initializer) as executor:
        futures = {executor.submit(analyzer, chunk) for chunk in sentence_chunks}
        for i, future in enumerate(as_completed(futures), 1):
            sys.stdout.write(f"\r  Processing text chunks: {i}/{len(sentence_chunks)}")
            sys.stdout.flush()
            chunk_words, chunk_grammar, chunk_tokenized_sents = future.result()
            word_counts.update(chunk_words)
            grammar.update(chunk_grammar)
            all_tokenized_sentences.extend(chunk_tokenized_sents)

    print("\n  Parallel analysis complete.")
    return dict(word_counts), grammar, all_tokenized_sentences


# --- i+1 Sentence Finding Logic ---
def find_i_plus_one_sentences(vocabulary_list: List[str], all_sentences: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Finds example sentences using a more intelligent sequential i+1 logic.
    """
    print("\n--- Pre-processing all sentences for faster lookups ---")
    word_to_sentences = defaultdict(list)
    for sentence_data in tqdm(all_sentences, desc="Building word index"):
        words_in_sentence = set(sentence_data["tokens"])
        for token in words_in_sentence:
            word_to_sentences[token].append(sentence_data)
            
    print("\n--- Analyzing vocabulary with smarter sequential i+1 logic ---")
    final_results = defaultdict(list)
    total_vocab_set = set(vocabulary_list)
    known_words = set()

    for current_word in tqdm(vocabulary_list, desc="Analyzing vocabulary"):
        relevant_sentences = word_to_sentences.get(current_word, [])
        
        for sentence_data in relevant_sentences:
            words_in_sentence_set = set(sentence_data["tokens"])

            if current_word in words_in_sentence_set:
                # --- NEW LOGIC ---
                # An "unknown" word is now defined as a word that is ALSO in our total vocab list
                # but has not yet been processed (i.e., it's a less frequent word).
                # This prevents penalizing sentences for containing random words not on our study list.
                future_vocab_words = words_in_sentence_set.intersection(total_vocab_set - known_words - {current_word})
                i_plus_one_score = len(future_vocab_words)

                if i_plus_one_score <= I_PLUS_ONE_THRESHOLD:
                    final_results[current_word].append({
                        "sentence": sentence_data["sentence"],
                        "i_plus_one_score": i_plus_one_score,
                        "source": sentence_data["source"]
                    })
        
        # After processing all sentences for the current word, add it to the set of known words.
        known_words.add(current_word)
        
    print("\n--- Sorting and selecting the best sentences for each word ---")
    sorted_results = {}
    for word in vocabulary_list:
        if word in final_results:
            # Sort the found sentences by their score (best score is 0, then 1, etc.)
            sorted_sentences = sorted(final_results[word], key=lambda x: x['i_plus_one_score'])
            # Limit the number of sentences to the configured amount
            sorted_results[word] = sorted_sentences[:NUM_EXAMPLE_SENTENCES]
            
    return sorted_results


# --- Parallel Dictionary Loading & Definition Adding ---
def get_dictionary_priority(folder_path: str) -> List[str]:
    """Prompts the user to select and prioritize dictionary files."""
    if not os.path.exists(folder_path):
        print(f"Warning: Dictionary folder '{folder_path}' not found.")
        return []
    dictionaries = [f for f in os.listdir(folder_path) if f.endswith('.zip') or f.endswith('.json')]
    if not dictionaries:
        print(f"No dictionary files found in the '{folder_path}' folder.")
        return []

    print("\nFound the following dictionaries:")
    for i, name in enumerate(dictionaries):
        print(f"  [{i + 1}] {name}")

    while True:
        try:
            priority_input = input(f"Enter the desired order (e.g., '2,1'), or press Enter to skip: ")
            if not priority_input:
                return []
            priority_indices = [int(p.strip()) - 1 for p in priority_input.split(',')]
            if all(0 <= i < len(dictionaries) for i in priority_indices):
                return [dictionaries[i] for i in priority_indices]
            else:
                print("Invalid input. Please enter numbers corresponding to the dictionaries listed.")
        except (ValueError, IndexError):
            print("Invalid input format. Please enter comma-separated numbers (e.g., 2,1).")

def load_single_dictionary_job(filepath: str) -> Tuple[str, Dict[str, Any]]:
    """Loads a single dictionary file (zip or json) into memory."""
    dict_name = os.path.splitext(os.path.basename(filepath))[0]
    dictionary = {}
    def process_entry(entry):
        try:
            term, _, _, _, _, definition, *_ = entry
            if isinstance(definition, (list, dict)):
                return term, orjson.dumps(definition).decode('utf-8')
            return term, definition
        except (ValueError, TypeError):
            return None, None
    try:
        if filepath.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as z:
                for json_file in z.namelist():
                    if json_file.startswith('term_bank_') and json_file.endswith('.json'):
                        with z.open(json_file) as f:
                            data = orjson.loads(f.read())
                            for entry in data:
                                term, definition = process_entry(entry)
                                if term: dictionary[term] = definition
        elif filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = orjson.loads(f.read())
                for entry in data:
                    term, definition = process_entry(entry)
                    if term: dictionary[term] = definition
    except Exception as e:
        print(f"\nERROR: Failed to load dictionary '{filepath}': {e}")
        return dict_name, {}
    return dict_name, dictionary

def load_dictionaries_with_cache(folder_path: str, prioritized_list: List[str]) -> Tuple[Dict[str, Dict], List[str]]:
    """Loads dictionaries from source or cached .pkl files for speed."""
    dictionaries = {}
    cached_paths = []
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    files_to_load_from_source = []

    for filename in prioritized_list:
        source_path = os.path.join(folder_path, filename)
        dict_name = os.path.splitext(filename)[0]
        cache_path = os.path.join(CACHE_FOLDER, f"{dict_name}.pkl")
        cached_paths.append(cache_path)

        is_stale = not os.path.exists(cache_path) or os.path.getmtime(source_path) > os.path.getmtime(cache_path)

        if is_stale:
            files_to_load_from_source.append(source_path)
        else:
            print(f"  Loading '{dict_name}' from cache...")
            try:
                with open(cache_path, 'rb') as f:
                    dictionaries[dict_name] = pickle.load(f)
            except Exception as e:
                print(f"  ERROR: Failed to load cache for '{dict_name}': {e}. Loading from source.")
                files_to_load_from_source.append(source_path)

    if files_to_load_from_source:
        print("Loading dictionaries from source files (and creating/updating cache)...")
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(load_single_dictionary_job, filepath) for filepath in files_to_load_from_source}
            for i, future in enumerate(as_completed(futures), 1):
                dict_name, loaded_dict = future.result()
                dictionaries[dict_name] = loaded_dict
                sys.stdout.write(f"\r  Loaded {i}/{len(files_to_load_from_source)}: '{dict_name}' ({len(loaded_dict)} entries)")
                sys.stdout.flush()
                cache_file = os.path.join(CACHE_FOLDER, f"{dict_name}.pkl")
                with open(cache_file, 'wb') as f:
                    pickle.dump(loaded_dict, f)
        print("\nDictionaries loaded and cached.")

    ordered_dicts = {name: dictionaries[name] for name in [os.path.splitext(p)[0] for p in prioritized_list] if name in dictionaries}
    return ordered_dicts, cached_paths

worker_dictionaries = {}
def initialize_definition_worker_from_cache(cached_file_paths: List[str]):
    """Initializer for each worker process to load dictionaries from .pkl cache files."""
    global worker_dictionaries
    for path in cached_file_paths:
        try:
            dict_name = os.path.splitext(os.path.basename(path))[0]
            with open(path, 'rb') as f:
                worker_dictionaries[dict_name] = pickle.load(f)
        except Exception as e:
            print(f"\nWorker process failed to load cache file {path}: {e}")

def find_definitions_for_chunk(word_chunk: List[str], priority_order: List[str]) -> Dict[str, str]:
    """Worker function to find definitions using its pre-loaded global dictionaries."""
    results = {}
    for word in word_chunk:
        definitions = []
        for dict_key in priority_order:
            if dict_key in worker_dictionaries and word in worker_dictionaries[dict_key]:
                definitions.append(f"<b>{dict_key}:</b> {worker_dictionaries[dict_key][word]}")
        if definitions:
            results[word] = "".join(definitions)
    return results

def add_definitions_parallel(words: Dict[str, Dict], prioritized_list: List[str], cached_paths: List[str]) -> Dict[str, Dict]:
    """Adds definitions to words in parallel to handle large datasets efficiently."""
    print("Adding definitions to words (in parallel)...")
    if not prioritized_list:
        print("  No dictionaries selected, skipping.")
        return words

    word_list = list(words.keys())
    chunk_size = 500
    word_chunks = [word_list[i:i + chunk_size] for i in range(0, len(word_list), chunk_size)]
    processed_count = 0
    total_chunks = len(word_chunks)
    priority_keys = [os.path.splitext(p)[0] for p in prioritized_list]

    with ProcessPoolExecutor(initializer=initialize_definition_worker_from_cache, initargs=(cached_paths,)) as executor:
        futures = {executor.submit(find_definitions_for_chunk, chunk, priority_keys) for chunk in word_chunks}
        for future in as_completed(futures):
            chunk_results = future.result()
            for word, definition in chunk_results.items():
                words[word]['definitions'] = definition
            processed_count += 1
            sys.stdout.write(f"\r  Processed definition chunks: {processed_count}/{total_chunks}")
            sys.stdout.flush()
    print("\nDefinitions added.")
    return words

# --- Filtering and Output ---
def interactive_filter(words: Dict[str, Dict]) -> Dict[str, Dict]:
    """Applies a series of user-defined filters to the word list."""
    print("\n--- Interactive Filtering ---")
    if not words:
        print("  Word list is empty, nothing to filter.")
        return {}

    if input("Remove words without any definitions? (y/n): ").lower() == 'y':
        words = {w: d for w, d in words.items() if d.get('definitions')}
        print(f"  Words remaining: {len(words)}")

    try:
        min_freq = int(input("Enter minimum appearance count (e.g., 5, or 0 for no limit): "))
        if min_freq > 0:
            words = {w: d for w, d in words.items() if d['count'] >= min_freq}
            print(f"  Words remaining: {len(words)}")
    except ValueError:
        print("  Invalid number, skipping frequency filter.")

    try:
        sorted_words = sorted(words.items(), key=lambda item: item[1]['count'], reverse=True)
        top_n = int(input(f"Show how many top common words for manual exclusion? (e.g., 20): "))
        if top_n > 0:
            print("Top common words:")
            to_exclude_preview = sorted_words[:min(top_n, len(sorted_words))]
            for word, data in to_exclude_preview:
                print(f"  - {word} (appears {data['count']} times)")
            if input("Filter out these top N words? (y/n): ").lower() == 'y':
                to_remove = {word for word, data in to_exclude_preview}
                words = {w: d for w, d in words.items() if w not in to_remove}
                print(f"  Words remaining: {len(words)}")
    except ValueError:
        print("  Invalid number, skipping common word filter.")

    manual_exclude = input("Enter any other words to exclude (comma-separated): ")
    if manual_exclude:
        to_remove = {w.strip() for w in manual_exclude.split(',')}
        words = {w: d for w, d in words.items() if w not in to_remove}
        print(f"  Words remaining: {len(words)}")
    return words

def generate_anki_deck(words: Dict[str, Any], filename: str, language: str):
    """Generates a TSV file for Anki import, highlighting all word occurrences."""
    okt = None
    if language == 'korean':
        try:
            from konlpy.tag import Okt
            okt = Okt()
        except ImportError:
            print("Warning: KoNLPy not found. Korean word stemming will be disabled.")

        print("  Pre-processing sentences for Korean highlighting...")
        sentence_cache = {}
        all_sentences = set()
        for data in words.values():
            # Handle cases where sentences might be missing for a word
            for s_data in data.get('sentences', []):
                all_sentences.add(s_data['sentence'])

        for i, sentence in enumerate(list(all_sentences)):
            sys.stdout.write(f"\r    Tokenizing sentence {i+1}/{len(all_sentences)}")
            sys.stdout.flush()
            try:
                tokens_original = okt.pos(sentence, norm=True, stem=False)
                tokens_stemmed = okt.pos(sentence, norm=True, stem=True)
                sentence_cache[sentence] = (tokens_original, tokens_stemmed)
            except Exception as e:
                print(f"\nWarning: KoNLPy failed on sentence: '{sentence}'. Error: {e}")
                sentence_cache[sentence] = None
        print("\n  Sentence processing complete.")


    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        sorted_words = sorted(words.items(), key=lambda item: item[1]['count'], reverse=True)
        
        for word, data in sorted_words:
            sentences = data.get('sentences', [])
            highlighted_sentences = []

            for sentence_data in sentences:
                sentence = sentence_data['sentence']
                highlighted_sentence = sentence
                konlpy_highlighted = False

                if language == 'korean' and okt and sentence in sentence_cache and sentence_cache[sentence]:
                    try:
                        tokens_original, tokens_stemmed = sentence_cache[sentence]
                        stem_to_original_map = defaultdict(set)
                        for (original_token, _), (stemmed_token, _) in zip(tokens_original, tokens_stemmed):
                            stem_to_original_map[stemmed_token].add(original_token)

                        forms_to_highlight = stem_to_original_map.get(word, set())

                        if forms_to_highlight:
                            forms = sorted(list(forms_to_highlight), key=len, reverse=True)
                            pattern_str = '|'.join(re.escape(f) for f in forms)
                            # --- FIX: Use a raw string to avoid SyntaxWarning ---
                            replacement = r'<span style="color: #4285F4;"><b><i>\g<0></i></b></span>'
                            highlighted_sentence = re.sub(pattern_str, replacement, sentence, flags=re.IGNORECASE)
                            konlpy_highlighted = True
                    except Exception as e:
                        print(f"Warning: KoNLPy highlighting failed on '{sentence}'. Falling back. Error: {e}")

                if not konlpy_highlighted:
                    boundary = r'\b' if language in ['spanish'] else ''
                    pattern = boundary + re.escape(word) + boundary
                    replacement = f'<span style="color: #4285F4;"><b><i>{word}</i></b></span>'
                    highlighted_sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)

                highlighted_sentences.append(f'<li>{highlighted_sentence}</li>')

            formatted_sentences = '<ul>' + ''.join(highlighted_sentences) + '</ul>'
            formatted_definition = parse_definition(data.get('definitions', ''))
            writer.writerow([word, formatted_definition, formatted_sentences, data.get('count', 0)])

# --- Main Execution Logic ---
def process_file_for_data(file_path: str, language: str) -> Tuple[Dict, Counter, List[Dict]]:
    """Orchestrates the analysis pipeline for a single file, returning all necessary data."""
    print(f"\n--- Processing file: {os.path.basename(file_path)} ---")
    file_extension = os.path.splitext(file_path)[1].lower()
    text = extract_text_from_epub(file_path) if file_extension == '.epub' else extract_text_from_txt(file_path)
    if not text:
        return {}, Counter(), []
    words, grammar, tokenized_sentences = analyze_text_parallel(text, language, file_path)
    print(f"  Found {len(words)} unique words and {len(tokenized_sentences)} valid sentences.")
    return words, grammar, tokenized_sentences

def main():
    """Main execution function."""
    while True:
        lang_choice = input("Select language (1 for Korean, 2 for Japanese, 3 for Spanish): ").strip()
        if lang_choice == '1':
            language = 'korean'
            break
        elif lang_choice == '2':
            language = 'japanese'
            break
        elif lang_choice == '3':
            language = 'spanish'
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    print(f"Language set to: {language.capitalize()}")

    source_files = find_source_files()
    if not source_files:
        print("Error: No .epub or .txt files found in '.', 'input/', or 'novels/' directories.")
        return

    selected_files = []
    if len(source_files) > 1:
        print("\nMultiple files found:")
        for i, filename in enumerate(source_files, 1):
            print(f"  [{i}] {filename}")
        choice = input("Enter 'all' or comma-separated numbers (e.g., '1,3') to process: ").lower()
        if choice == 'all':
            selected_files = source_files
        else:
            try:
                indices = [int(i.strip()) - 1 for i in choice.split(',')]
                selected_files = [source_files[i] for i in indices if 0 <= i < len(source_files)]
            except (ValueError, IndexError):
                print("Invalid input. Exiting.")
                return
    else:
        selected_files = source_files

    if not selected_files:
        print("No files selected. Exiting.")
        return

    all_words_combined = Counter()
    all_tokenized_sentences_combined = []
    
    for file_path in selected_files:
        try:
            word_counts, _, tokenized_sentences = process_file_for_data(file_path, language)
            all_words_combined.update(word_counts)
            all_tokenized_sentences_combined.extend(tokenized_sentences)
        except Exception as e:
            print(f"\n--- !!! ---")
            print(f"ERROR: An unexpected error occurred while processing '{os.path.basename(file_path)}'.")
            print(f"The error was: {e}")
            print("The script will attempt to continue with the next file.")
            print(f"--- !!! ---\n")

    if not all_words_combined:
        print("No words were extracted from the selected files. Exiting.")
        return

    print(f"\n--- Combined Processing for All Files ---")
    print(f"Total unique words found: {len(all_words_combined)}")
    print(f"Total valid sentences collected: {len(all_tokenized_sentences_combined)}")

    final_word_data = {word: {'count': count} for word, count in all_words_combined.items()}

    prioritized_dictionaries = get_dictionary_priority(DICTIONARIES_FOLDER)
    if prioritized_dictionaries:
        _, cached_dictionary_paths = load_dictionaries_with_cache(DICTIONARIES_FOLDER, prioritized_dictionaries)
        final_word_data = add_definitions_parallel(final_word_data, prioritized_dictionaries, cached_dictionary_paths)

    final_word_data = interactive_filter(final_word_data)
    
    try:
        limit_input = input(f"Enter the total number of words for the final deck (e.g., 2000, or press Enter to keep all): ")
        if limit_input.strip():
            limit = int(limit_input)
            if limit > 0 and limit < len(final_word_data):
                print(f"  Limiting final deck to the top {limit} most frequent words.")
                sorted_words = sorted(final_word_data.items(), key=lambda item: item[1]['count'], reverse=True)
                final_word_data = dict(sorted_words[:limit])
                print(f"  Words remaining: {len(final_word_data)}")
    except ValueError:
        print("  Invalid number. Keeping all remaining words.")


    if not final_word_data:
        print("All words were filtered out. No Anki deck will be generated. Exiting.")
        return

    # Create the frequency-sorted vocabulary list for the i+1 logic.
    sorted_word_items = sorted(final_word_data.items(), key=lambda item: item[1]['count'], reverse=True)
    vocabulary_list_for_i_plus_one = [word for word, data in sorted_word_items]

    i_plus_one_sentences = find_i_plus_one_sentences(vocabulary_list_for_i_plus_one, all_tokenized_sentences_combined)
    
    for word, sentences in i_plus_one_sentences.items():
        if word in final_word_data:
            final_word_data[word]['sentences'] = sentences

    os.makedirs(ANKI_FOLDER, exist_ok=True)
    combined_anki_file = os.path.join(ANKI_FOLDER, 'combined_anki_deck.csv')
    print(f"\n--- Generating final combined Anki deck ---")
    generate_anki_deck(final_word_data, combined_anki_file, language)
    print(f"  - Combined Anki deck with i+1 sentences saved to: {combined_anki_file}")

    print(f"\nAnalysis complete!")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()