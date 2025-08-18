import csv
import json
import re

def parse_json_to_html(json_data):
    """
    사용자 정의 JSON 구조를 재귀적으로 파싱하여 유효한 HTML 문자열을 생성합니다.
    """
    html_parts = []
    
    # 일관된 처리를 위해 단일 객체를 리스트로 래핑
    items = json_data if isinstance(json_data, list) else [json_data]
    
    for item in items:
        # 항목이 문자열인 경우, 직접 추가
        if isinstance(item, str):
            html_parts.append(item)
            continue
            
        tag = item.get('tag')
        content = item.get('content')
        style = item.get('style', {})
        lang = item.get('lang', None)

        # 스타일 딕셔너리를 유효한 HTML 스타일 문자열로 변환
        style_attrs = ''
        if style:
            style_str_list = []
            for key, value in style.items():
                # camelCase를 kebab-case로 변환
                kebab_key = re.sub(r'(?<!^)(?=[A-Z])', '-', key).lower()
                style_str_list.append(f'{kebab_key}:{value};')
            style_attrs = f' style="{"".join(style_str_list)}"'
        
        # 다른 속성들을 구성
        lang_attr = f' lang="{lang}"' if lang else ''
        attributes = f'{style_attrs}{lang_attr}'
        
        # 'tag' 필드가 존재할 때만 새 태그를 생성
        if tag:
            # HTML 태그 시작
            html_parts.append(f'<{tag}{attributes}>')

        # 재귀 호출을 통해 중첩된 콘텐츠 처리
        if content:
            if isinstance(content, (list, dict)):
                html_parts.append(parse_json_to_html(content))
            else:
                html_parts.append(str(content))
        
        # 태그가 열렸을 때만 태그 닫기
        if tag:
            # HTML 태그 닫기
            html_parts.append(f'</{tag}>')
    
    return ''.join(html_parts)

def main():
    """
    CSV를 읽고 데이터를 파싱하여 새 CSV를 작성하는 메인 함수입니다.
    """
    input_file = 'input.csv'
    output_file = 'output.csv'

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            
            reader = csv.reader(infile, delimiter='\t')
            writer = csv.writer(outfile, delimiter='\t')
            
            # 헤더 읽기 및 쓰기
            header = next(reader)
            writer.writerow(header)
            
            # CSV의 각 행 처리
            for i, row in enumerate(reader):
                if len(row) < 2:
                    print(f"Skipping malformed row {i+1}.")
                    writer.writerow(row)
                    continue

                original_string = row[1]
                
                # 정규 표현식으로 JSON 문자열을 더 안정적으로 추출
                match = re.search(r'\[\{.*\}\]\s*</div>', original_string, re.DOTALL)
                
                if not match:
                    writer.writerow(row)
                    continue

                try:
                    # 일치하는 부분만 추출
                    json_with_trailing_div = match.group(0)
                    # 끝에 있는 </div> 제거
                    json_string_raw = json_with_trailing_div[:-6].strip()
                    
                    # CSV 이스케이프된 이중 따옴표를 유효한 JSON 형식으로 수정
                    corrected_json_string = json_string_raw.replace('""', '"')
                    
                    # 수정된 JSON 문자열을 파이썬 객체로 변환
                    parsed_json = json.loads(corrected_json_string)
                    
                    # 파이썬 객체를 새 HTML 문자열로 변환
                    parsed_html_content = parse_json_to_html(parsed_json)
                    
                    # 출력 파일에 대한 전체 HTML 문자열 재구성
                    final_html = f'<div class="definition-content"><b>KO-EN.KRDICT:</b> {parsed_html_content}</div>'
                    
                    # 행을 새 HTML 문자열로 업데이트
                    new_row = row[:]
                    new_row[1] = final_html
                    
                    # 처리된 행을 출력 파일에 쓰기
                    writer.writerow(new_row)
                    
                    if (i + 1) % 1000 == 0:
                        print(f"Processed {i+1} entries...")
                        
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error decoding JSON in row {i+1}: {e}. Skipping conversion for this row.")
                    writer.writerow(row)
                
            print(f"Processing complete. Output saved to '{output_file}'.")
            
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()