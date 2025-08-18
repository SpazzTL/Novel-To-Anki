import csv
import json
import re

# --- Configuration ---
# Set the desired number of entries to process.
# This will keep the top 'LIMIT' most popular words from your sorted CSV.
LIMIT = 5000

# --- JSON to HTML Conversion Function ---
def parse_json_to_html(json_data):
    """
    Recursively parses the custom JSON structure to build a valid HTML string.
    """
    html_parts = []
    
    # Wrap single objects in a list for consistent processing
    items = json_data if isinstance(json_data, list) else [json_data]
    
    for item in items:
        # If the item is a string, just add it directly.
        if isinstance(item, str):
            html_parts.append(item)
            continue
            
        tag = item.get('tag')
        content = item.get('content')
        style = item.get('style', {})
        lang = item.get('lang', None)

        # Convert style dictionary to a valid HTML style string
        style_attrs = ''
        if style:
            style_str_list = []
            for key, value in style.items():
                kebab_key = re.sub(r'(?<!^)(?=[A-Z])', '-', key).lower()
                style_str_list.append(f'{kebab_key}:{value};')
            style_attrs = f' style="{"".join(style_str_list)}"'
        
        # Build other attributes
        lang_attr = f' lang="{lang}"' if lang else ''
        attributes = f'{style_attrs}{lang_attr}'
        
        # Only create a new tag if the 'tag' field exists.
        if tag:
            html_parts.append(f'<{tag}{attributes}>')

        # Handle nested content by making a recursive call
        if content:
            if isinstance(content, (list, dict)):
                html_parts.append(parse_json_to_html(content))
            else:
                html_parts.append(str(content))
        
        # Only close the tag if a tag was opened.
        if tag:
            html_parts.append(f'</{tag}>')
    
    return ''.join(html_parts)

# --- Main Processing Logic ---
def main():
    """
    Main function to read the CSV, perform JSON-to-HTML conversion,
    and limit the output to a specified number of entries.
    """
    input_file = 'output.csv'
    output_file = 'limited_output.csv'

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            
            reader = csv.reader(infile, delimiter='\t')
            writer = csv.writer(outfile, delimiter='\t')
            
            # Read and write the header
            header = next(reader)
            writer.writerow(header)
            
            # Process and write only the first 'LIMIT' rows
            for i, row in enumerate(reader):
                if i >= LIMIT:
                    print(f"Reached the limit of {LIMIT} entries. Stopping.")
                    break

                if len(row) < 2:
                    print(f"Skipping malformed row {i+1}.")
                    writer.writerow(row)
                    continue

                original_string = row[1]
                
                # Check if the string contains the expected JSON structure
                if 'KO-EN.KRDICT: [' not in original_string and 'KO-EN.KRDICT: <span' not in original_string:
                    writer.writerow(row)
                    continue

                try:
                    if 'KO-EN.KRDICT: [' in original_string:
                        # Case for JSON string
                        match = re.search(r'\[\{.*\}\]\s*</div>', original_string, re.DOTALL)
                        
                        if not match:
                            writer.writerow(row)
                            continue

                        json_with_trailing_div = match.group(0)
                        json_string_raw = json_with_trailing_div[:-6].strip()
                        corrected_json_string = json_string_raw.replace('""', '"')
                        parsed_json = json.loads(corrected_json_string)
                        parsed_html_content = parse_json_to_html(parsed_json)
                        final_html = f'<div class="definition-content"><b>KO-EN.KRDICT:</b> {parsed_html_content}</div>'

                    else:
                        # Case for already processed HTML
                        final_html = original_string

                    # Update the row with the new HTML string
                    new_row = row[:]
                    new_row[1] = final_html
                    
                    # Write the processed row to the output file
                    writer.writerow(new_row)
                    
                    if (i + 1) % 1000 == 0:
                        print(f"Processed {i+1} entries...")
                        
                except (json.JSONDecodeError, ValueError, IndexError) as e:
                    print(f"Error decoding JSON in row {i+1}: {e}. Skipping conversion for this row.")
                    writer.writerow(row)
                
            print(f"Processing complete. Output saved to '{output_file}'.")
            
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()