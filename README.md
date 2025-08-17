


# Novel-To-Anki
Inspects .epub and .txt files to find unique words, sentences in which they are contained, their definitions from user provided dictionaries, and generates a JSON and CSV file, with the CSV being formatted for ANKI.
Example json output:
``` 
"입술": {
    "count": 57,
    "sentences": [
      "눈썹, 눈 코 입술.",
      "날카로운 눈매, 탐욕스러워 보이는 입술.",
      "움푹 팬 눈, 핏기 없는 입술.",
      "그저 입술을 파들파들 떨 뿐이었다.",
      "단어들이 그녀의 입술에서 아무렇게나 쏟아져 나왔다."
    ],
    "definitions": "<b>krdict_v2:</b> ['lips']"
  },
  "천천히": {
    "count": 80,
    "sentences": [
      "천천히, 꼼꼼하게.",
      "그저 내 등을 가만히, 천천히 토닥여줄 뿐이었다.",
      "그리고 내게 천천히 다가왔다.",
      "이렇게까지 될 일은 없었을 거야.\" 그녀의 목소리가 천천히 커져갔다.",
      "해가 천천히 서쪽으로 기울고, 하늘이 주황빛으로 물들기 시작할 무렵이었다."
    ],
    "definitions": "<b>krdict_v2:</b> ['slowly', 'slowly']"
  } 
```


Only tested with korean novels, might also work for other languages.
---

# How to use:
Place the script in a folder with the EPUB | TXT files you want to parse. Create a folder named **Dictionaries** and place zipped (yomitan!) dictionaries inside it.
(You can also have a folder called `novels` or `input` with all txt / epubs.)

[Example Dictionaries](https://github.com/Lyroxide/yomitan-ko-dic/releases)

**File Example:**

-   **Root**
    -   epub-anki.py
    -   Novel1.epub
    -   Novel2.epub (Optional)
    -   **Dictionaries/** (Folder)
        -   Dictionary1.zip
        -   Dictionary2.zip

A folder named **cache** will be generated to load dictionaries faster.

A folder named **output** will be generated to store the output.

## Requirements

* Python 3.7+
* * Python libraries: ebooklib, konlpy, orjson(optional)

# Anki Formatting 
A .apkg is attached with styling info
<img width="899" height="730" alt="image" src="https://github.com/user-attachments/assets/1f272170-8d6f-42a0-aeb7-ac730db7ce54" />
<img width="930" height="488" alt="image" src="https://github.com/user-attachments/assets/8e62c5c4-f63b-4757-915b-d8b92b730207" />
