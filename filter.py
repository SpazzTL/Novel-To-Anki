import csv
import os
import glob
from collections import defaultdict
import sys
from itertools import islice

def find_csv_files(relative_path='output/anki_csv'):
    """
    Finds all CSV files in the script's directory and a specified relative path.
    """
    csv_files = []
    csv_files.extend(glob.glob('*.csv'))
    if os.path.isdir(relative_path):
        csv_files.extend(glob.glob(os.path.join(relative_path, '*.csv')))
    return sorted(list(set(csv_files)))

def get_word_frequencies(input_file):
    """
    Reads a CSV file and returns a dictionary of word frequencies.
    """
    all_words = defaultdict(int)
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) > 1:
                    word = row[0].strip()
                    all_words[word] += 1
    except FileNotFoundError:
        print(f"\nError: The file '{input_file}' was not found.")
        return None
    return all_words

def get_user_input(prompt, type_func, default_value, validation_func=None):
    """
    Prompts the user for input with a default value and performs type conversion and validation.
    """
    while True:
        user_input = input(f"{prompt} (Default: {default_value}): ")
        if not user_input:
            return default_value
        try:
            value = type_func(user_input)
            if validation_func and not validation_func(value):
                print("Invalid input. Please try again.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a value of the correct type.")

def get_interactive_word_list(sorted_words, prompt_help, prompt_header):
    """
    Displays a paginated, interactive list of words with frequencies for user selection.
    
    Args:
        sorted_words (list): A list of tuples (word, frequency), sorted by frequency.
        prompt_help (str): Help text for the user.
        prompt_header (str): Header to display before the list.
    """
    page_size = 25
    selected_words = set()
    current_page = 0
    total_words = len(sorted_words)
    total_pages = (total_words + page_size - 1) // page_size
    
    def display_page(page):
        start = page * page_size
        end = start + page_size
        print(prompt_header)
        for i, (word, freq) in enumerate(islice(sorted_words, start, end)):
            prefix = "[x]" if word in selected_words else "[ ]"
            print(f"{prefix} [{start + i + 1:04}] {word:<20} [{freq}]")
        print(f"\nPage {page + 1}/{total_pages} | Total words: {total_words}")
        print("--- Navigation ---")
        print("n: next page, b: back a page")
        print("j [num]: jump to page number, f [num]: jump to frequency")
        print("s [num] or s [num1-num2, ...]: select/deselect item(s), d: finish selection")
        print("h: help")
        
    while True:
        display_page(current_page)
        user_input = input("\nEnter a command: ").strip().lower()
        
        if user_input == 'n':
            if current_page < total_pages - 1:
                current_page += 1
            else:
                print("You are on the last page.")
        elif user_input == 'b':
            if current_page > 0:
                current_page -= 1
            else:
                print("You are on the first page.")
        elif user_input.startswith('j '):
            try:
                page_num = int(user_input.split(' ')[1])
                if 1 <= page_num <= total_pages:
                    current_page = page_num - 1
                else:
                    print(f"Invalid page number. Please enter a number between 1 and {total_pages}.")
            except (ValueError, IndexError):
                print("Invalid jump command. Use 'j [page_number]'.")
        elif user_input.startswith('f '):
            try:
                freq = int(user_input.split(' ')[1])
                first_match_index = -1
                for i, (word, word_freq) in enumerate(sorted_words):
                    if word_freq <= freq:
                        first_match_index = i
                        break
                
                if first_match_index != -1:
                    current_page = first_match_index // page_size
                else:
                    print(f"No words found with frequency <= {freq}.")
            except (ValueError, IndexError):
                print("Invalid frequency command. Use 'f [frequency_number]'.")
        elif user_input.startswith('s '):
            selections = user_input[2:].strip().split(',')
            for selection in selections:
                selection = selection.strip()
                try:
                    if '-' in selection:
                        start_str, end_str = selection.split('-')
                        start = int(start_str)
                        end = int(end_str)
                        
                        if start > end:
                            print(f"Invalid range: {start}-{end}. Start number must be less than or equal to end number.")
                            continue
                            
                        if not (1 <= start <= total_words and 1 <= end <= total_words):
                            print(f"Invalid range: {start}-{end}. Numbers must be between 1 and {total_words}.")
                            continue
                            
                        for index in range(start - 1, end):
                            word_to_toggle = sorted_words[index][0]
                            if word_to_toggle in selected_words:
                                selected_words.remove(word_to_toggle)
                            else:
                                selected_words.add(word_to_toggle)
                        print(f"Toggled selection for words {start} through {end}.")
                    else:
                        index = int(selection)
                        if 1 <= index <= total_words:
                            word_to_toggle = sorted_words[index - 1][0]
                            if word_to_toggle in selected_words:
                                selected_words.remove(word_to_toggle)
                                print(f"'{word_to_toggle}' unselected.")
                            else:
                                selected_words.add(word_to_toggle)
                                print(f"'{word_to_toggle}' selected.")
                        else:
                            print(f"Invalid item number: {index}. Please enter a number between 1 and {total_words}.")
                except (ValueError, IndexError):
                    print(f"Invalid selection format: '{selection}'. Use 's [num]' or 's [num1-num2]'.")
        elif user_input == 'd':
            return selected_words
        elif user_input == 'h':
            print(prompt_help)
        else:
            print("Invalid command. Please use 'n', 'b', 'j [num]', 'f [num]', 's [num]', 'd', or 'h'.")

def filter_anki_deck(input_file, output_file, config):
    """
    Filters an Anki deck CSV file based on a configuration dictionary.
    """
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\nCreated directory: {output_dir}")

    total_lines = 0
    filtered_data = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            reader = csv.reader(f_in, delimiter='\t')
            for row in reader:
                total_lines += 1
                if len(row) > 1:
                    word = row[0].strip()
                    
                    is_excluded_word = word in config['excluded_words']
                    is_name = any(name in row[2] for name in config['names'])
                    is_definition_name = len(row) > 1 and row[1].split()[0] in config['names']
                    is_too_long = len(word) > config['max_word_length']
                    is_too_short = len(word) < config['min_word_length']

                    if not any([is_excluded_word, is_name, is_definition_name, is_too_long, is_too_short]):
                        filtered_data.append(row)
    
    except FileNotFoundError:
        print(f"\nError: The file '{input_file}' was not found.")
        return
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out, delimiter='\t')
        writer.writerows(filtered_data)
    
    print(f"\nFiltering complete!")
    print(f"Original file had {total_lines} lines.")
    print(f"Filtered file has {len(filtered_data)} lines.")
    print(f"Filtered deck saved to {output_file}")
    
    return total_lines, len(filtered_data)

def run_interactive_filter():
    """
    Main function to run the script with a user-friendly, step-by-step flow.
    """
    available_files = find_csv_files()
    
    if not available_files:
        print("No CSV files found in the current directory or 'output/anki_csv'.")
        input("\nPress Enter to exit...")
        return
    
    print("--- Step 1: File Selection ---")
    print("Found the following CSV files:")
    for i, filename in enumerate(available_files):
        print(f"[{i+1}] {filename}")
    
    while True:
        try:
            choice = int(input("\nPlease enter the number of the file you want to filter: "))
            if 1 <= choice <= len(available_files):
                selected_file = available_files[choice - 1]
                break
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    word_frequencies = get_word_frequencies(selected_file)
    if not word_frequencies:
        input("\nPress Enter to exit...")
        return

    sorted_words = sorted(word_frequencies.items(), key=lambda item: item[1], reverse=True)
    
    print("\n--- Step 2: Interactive Word Filtering ---")
    print("Use the commands to navigate the list of words and their frequencies.")
    print("Select words you want to exclude from the final deck.")

    excluded_words = get_interactive_word_list(
        sorted_words,
        "Enter 's [number]' or 's [num1-num2, ...]' to select/deselect word(s). Use 'd' to finish.",
        "\n--- Word List (Sorted by Frequency) ---"
    )

    print("\n--- Step 3: Name Filtering ---")
    default_names = "은월, 하연, 한서진, 김재훈"
    names_str = input(f"Enter names to exclude (comma-separated) (Default: '{default_names}'): ")
    names_str = names_str if names_str else default_names
    names = {name.strip() for name in names_str.split(',')}

    print("\n--- Step 4: Word Length Filtering ---")
    min_word_length = get_user_input("Enter minimum word length to keep", int, 2, lambda x: x > 0)
    max_word_length = get_user_input("Enter maximum word length to keep", int, 10, lambda x: x >= min_word_length)
    
    print("\n--- Step 5: Review and Confirm Filters ---")
    print(f"File to filter: {selected_file}")
    print(f"Words to exclude: {len(excluded_words)} selected")
    print(f"Names to exclude: {', '.join(names)}")
    print(f"Word length range: {min_word_length} to {max_word_length}")
    
    confirmation = input("\nDo you want to proceed with filtering? (y/n): ")
    if confirmation.lower() != 'y':
        print("Filtering aborted.")
        input("\nPress Enter to exit...")
        return
        
    output_dir = os.path.join('output', 'anki_csv')
    base_name, ext = os.path.splitext(os.path.basename(selected_file))
    OUTPUT_FILE_NAME = os.path.join(output_dir, f'{base_name}_filtered{ext}')
    
    config = {
        'excluded_words': excluded_words,
        'names': names,
        'max_word_length': max_word_length,
        'min_word_length': min_word_length,
    }
    
    filter_anki_deck(selected_file, OUTPUT_FILE_NAME, config)
    
    input("\nPress Enter to exit...")

if __name__ == '__main__':
    run_interactive_filter()