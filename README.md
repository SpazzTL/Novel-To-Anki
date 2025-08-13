


# Novel-To-Anki
Inspects webnovels in .epub and .txt format and outputs a CSV file for Anki.
(only supports korean for now)
---

# How to use:
Place the script in a folder with the EPUB | TXT files you want to parse. Create a folder named **Dictionaries** and place zipped dictionaries inside it.
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

# Speed up:
run `pip install orjson'
