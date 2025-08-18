import re
from tokenize import tokenize, ENCODING, NEWLINE, ENDMARKER
from io import BytesIO
import json
import os

# List of Python keywords and commonly used function names
KEYWORDS = (
    "and", "as", "abs", "assert", "await", "async", "append",
    "break", "bin", "bool",
    "class", "continue", "clear", "count", "capitalize", "complex",
    "def", "del", "dict",
    "elif", "else", "exec", "end", "extend",
    "finally", "False", "for", "from", "find", "float", "functools", "filter",
    "get", "global",
    "hex", 
    "nonlocal",
    "if", "import", "in", "is", "input", "index", "isdigit", "insert", "int", "isupper", "islower", "items",
    "join",
    "keys",
    "lambda", "len", "lower", "list",
    "not", "None", "NEWLINE",
    "map",
    "or",
    "pass", "print", "pop",
    "raise", "return", "range", "round", "replace", "rfind", "reduce", "remove", "reverse",
    "str", "split", "sorted", "sort",
    "True", "tuple", "time", "type", "TABSPACE",
    "upper", "update",
    "values",
    "while", "SINGLE_QUOTE", "DOUBLE_QUOTE", "STRING", "RIGHT_PARENTHESIS",
    "STRING5", "STRING6", "STRING7", "NEWLINE0", "TABSPACE0", "COMMA", "EXCLAMATION", "LEFT_BRACKET", "RIGHT_BRACKET",
    "COLON", "input"
)

# Convert keywords to uppercase
UPPER_KEYWORDS = [x.upper() for x in KEYWORDS]

# Token patterns for lexical analysis
TOKENS = (
    (r'[a-zA-Z_]\w*', 'VAR'),
    (r'0', 'INT'),
    (r'[-+]?\d+[eE][-+]?\d+[jJ]', 'FLOAT_EXPONANT_COMPLEX'),
    (r'[-+]?\d+.\d?[eE][-+]?\d+[jJ]', 'FLOAT_EXPONANT_COMPLEX'),
    (r'[-+]?\d?.\d+[eE][-+]?\d+[jJ]', 'FLOAT_EXPONANT_COMPLEX'),
    (r'\d+[eE][-+]?\d*', 'FLOAT_EXPONANT'),
    (r'\d+\.\d*[eE][-+]?\d*', 'FLOAT_EXPONANT'),
    (r'\.\d+[eE][-+]?\d*', 'FLOAT_EXPONANT'),
    (r'\d*\.\d+[jJ]', 'COMPLEX'),
    (r'\d+\.[jJ]', 'COMPLEX'),
    (r'\d+[jJ]', 'COMPLEX'),
    (r'\d+\.', 'FLOAT'),
    (r'\d*[_\d]*\.[_\d]+[lL]?', 'FLOAT'),
    (r'\d+[_\d]+\.[_\d]*[lL]?', 'FLOAT'),
    (r'\.', 'DOT'),
    (r'[1-9]+[_\d]*[lL]', 'LONG'),
    (r'[1-9]+[_\d]*', 'INT'),
    (r'0[xX][\d_a-fA-F]+[lL]?', 'HEXA'),
    (r'(0[oO][0-7]+)|(0[0-7_]*)[lL]?', 'OCTA'),
    (r'0[bB][01_]+[lL]?', 'BINARY'),
    (r'\(', 'LEFT_PARENTHESIS'),
    (r'\)', 'RIGHT_PARENTHESIS'),
    (r':', 'COLON'),
    (r',', 'COMMA'),
    (r';', 'SEMICOLON'),
    (r'@', 'AT'),
    (r'\+', 'PLUS'),
    (r'-', 'MINUS'),
    (r'\*', 'STAR'),
    (r'/', 'SLASH'),
    (r'\|', 'VBAR'),
    (r'&', 'AMPER'),
    (r'@', 'AT'),
    (r'<', 'LESS'),
    (r'>', 'GREATER'),
    (r'=', 'EQUAL'),
    (r'%', 'PERCENT'),
    (r'\[', 'LEFT_SQUARE_BRACKET'),
    (r'\]', 'RIGHT_SQUARE_BRACKET'),
    (r'\{', 'LEFT_BRACKET'),
    (r'\}', 'RIGHT_BRACKET'),
    (r'`', 'BACKQUOTE'),
    (r'==', 'EQUAL_EQUAL'),
    (r'<>', 'NOT_EQUAL'),
    (r'!=', 'NOT_EQUAL'),
    (r'<=', 'LESS_EQUAL'),
    (r'>=', 'GREATER_EQUAL'),
    (r'~', 'TILDE'),
    (r'\^', 'CIRCUMFLEX'),
    (r'<<', 'LEFT_SHIFT'),
    (r'>>', 'RIGHT_SHIFT'),
    (r'\*\*', 'DOUBLE_STAR'),
    (r'\+=', 'PLUS_EQUAL'),
    (r'-=', 'MINUS_EQUAL'),
    (r'@=', 'AT_EQUAL'),
    (r'\*=', 'STAR_EQUAL'),
    (r'/=', 'SLASH_EQUAL'),
    (r'%=', 'PERCENT_EQUAL'),
    (r'&=', 'AMPER_EQUAL'),
    (r'\|=', 'VBAR_EQUAL'),
    (r'\^=', 'CIRCUMFLEX_EQUAL'),
    (r'<<=', 'LEFT_SHIFT_EQUAL'),
    (r'>>=', 'RIGHT_SHIFT_EQUAL'),
    (r'\.\.\.', 'ELLIPSIS'),
    (r'->', 'RIGHT_ARROW'),
    (r'\*\*=', 'DOUBLE_STAR_EQUAL'),
    (r'//', 'DOUBLE_SLASH'),
    (r'//=', 'DOUBLE_SLASH_EQUAL'),
    (r'[uU]["\'](.|\n|\r)*', 'UNICODE_STRING'),
    (r'[fF]["\'](.|\n|\r)*', 'INTERPOLATED_STRING'),
    (r'[rR]["\'](.|\n|\r)*', 'RAW_STRING'),
    (r'[bB]["\'](.|\n|\r)*', 'BINARY_STRING'),
    (r'[uU][rR]["\'](.|\n|\r)*', 'UNICODE_RAW_STRING'),
    (r'[bB][rR]["\'](.|\n|\r)*', 'BINARY_RAW_STRING'),
    (r'[fF][rR]["\'](.|\n|\r)*', 'INTERPOLATED_RAW_STRING'),
    (r'[rR][fF]["\'](.|\n|\r)*', 'INTERPOLATED_RAW_STRING'),
    (r'±', 'SUM_INTENDED'),
    (r'/n', 'endl')
)

TOKENS2 = []
for t in TOKENS:
    TOKENS2.append(t[1])
for t in UPPER_KEYWORDS:
    TOKENS2.append(t)

TOKENS2 = list(dict.fromkeys(TOKENS2))

TOKENS = [(re.compile('^' + x[0] + '$'), x[1]) for x in TOKENS]

listToPrint = (
    "degisken_ismi", "int", "float_deger", "float_deger", "complex_deger", "float_deger",
    ".", "long_deger", "hexa_deger", "octa_deger", "binary_deger",
    "(", ")", ": ", ",", ";",
    "@", "+", "-", "*", "\\",
    "vbar", "&", "<", ">", "=",
    "%", "[", "]", "{", "}",
    "'", "==", "!=", "<=", ">=",
    "~", "circumflex", "//", "\\", "**",
    "+=", "-=", "@=", "*=", "=",
    "%=", "&=", "vbar=", "circumflex=", "//=",
    "\\=", "ellipsis", "right_arrow", "**=", "///",
    "///=", "#comment", "string", "string", "string",
    "string", "string", "string", "string", "+=",
    "deneme", "", "and", "as", "abs",
    "abs", "assert", "await", "async", "append", "break",
    "bin", "bool", "class", "continue", "clear",
    "count", "capitalize", "def", "del", "dict",
    "elif", "else", "exec", "end", "extend",
    "finally", "False", "for", "from", "find",
    "functools", "filter", "get", "global", "hex", 
    "nonlocal", "if", "import", "in", "is",
    "input", "index", "isdigit", "insert", "isupper",
    "islower", "items", "join", "keys", "lambda",
    "len", "lower", "list", "not", "None",
    "\n", "map", "or", "pass", "print",
    "pop", "raise", "return", "range", "round",
    "replace", "rfind", "reduce", "remove", "reverse",
    "str", "split", "sorted", "sort", "True",
    "tuple", "time", "type", "    ", "upper",
    "update", "values", "while", "'", "\"",
    "string", "string", "string", "string", "\n",
    "    ", "!"
)

from tokenize import tokenize, ENCODING, NEWLINE, ENDMARKER, TokenInfo
from io import BytesIO

def more_tokenize(sequence, print_function=False):
    """
    Tokenizes the given sequence using the tokenize_generator function.
    """
    return list(tokenize_generator(sequence))

def tokenize_current_keywords(print_function=False):
    """
    Returns a list of keywords, excluding 'print' if print_function is True.
    """
    if print_function is True:
        return [x for x in KEYWORDS if x != "print"]
    else:
        return KEYWORDS

def listToString(lines):
    """
    Converts a list of strings into a single concatenated string.
    """
    str1 = ""
    for ele in lines:
        str1 += ele
    return str1

def tokenize_generator(sequence):
    """
    Generator function to tokenize a sequence based on predefined tokens.
    """
    current_keywords = tokenize_current_keywords()
    for item in sequence:
        if item in current_keywords:
            yield [item.upper(), item]
            continue
        
        for candidate, token_name in TOKENS:
            if candidate.match(item):
                yield [token_name, item]
                break
        else:
            yield [token_name, item]
            break

def call_moretokenizer(errorline):
    """
    Tokenizes a given line using Python's tokenize module and extracts token names and their indices.
    """
    g = tokenize(BytesIO(errorline.encode('utf-8')).readline)
    tokenarray = []
    for toknum, tokval, _, _, _ in g:
        if toknum not in [ENCODING, NEWLINE, ENDMARKER] and tokval != '':
            tokenarray.append(tokval)
    
    more_tkn = more_tokenize(tokenarray)
    tkn_named_entity = []
    tkn_ids = []
    for tkn in more_tkn:
        tkn_named_entity.append(tkn[0])
        tkn_ids.append(TOKENS2.index(tkn[0]))
    
    return tkn_named_entity, tkn_ids

def call_moretokenizer2(errorline):
    """
    Tokenizes the given error line and maps tokens to their respective positions.
    """
    from io import BytesIO
    from tokenize import tokenize, ENCODING, NEWLINE, ENDMARKER

    g = tokenize(BytesIO(errorline.encode('utf-8')).readline)
    tokenarray = []
    token_positions = []
    original_indices = []

    current_index = 0
    last_column = 0
    offset = 0

    for tok in g:
        toknum, tokval, start, end, _ = tok
        if toknum not in [ENCODING, NEWLINE, ENDMARKER] and tokval != '':
            tokenarray.append(tokval)
            token_positions.append((start, end))

            start_index = current_index + errorline[current_index:].find(tokval) + offset
            end_index = start_index + len(tokval)
            original_indices.append((start_index, end_index))
            current_index = end_index

    more_tkn = more_tokenize(tokenarray)  # Further token processing.
    tkn_named_entity = []
    tkn_ids = []
    tkn_original_positions = []
    satir_no = 1  # Line number tracking.
    prev_end_pos = 0

    for idx, tkn in enumerate(more_tkn):
        tkn_named_entity.append(tkn[0])
        tkn_id = TOKENS2.index(tkn[0])  # Convert token name to index.
        tkn_ids.append(tkn_id)
        start_pos_original, end_pos_original = original_indices[idx]

        if idx < len(more_tkn) - 1:
            space_between_tokens = original_indices[idx + 1][0] - end_pos_original
        else:
            space_between_tokens = 0

        start_pos = prev_end_pos
        
        if tkn_id in [121, 155]:  # Handles new line tokens.
            satir_no += 1
            start_pos = 0
            end_pos = end_pos_original
            prev_end_pos = end_pos
            end_pos = 0
        elif tkn_id in [150, 11, 12, 13, 14, 27, 28, 29, 30]:
            end_pos = start_pos + 1  # Single character tokens.
        elif tkn_id in [144, 156]:
            end_pos = start_pos + 4  # Tokens requiring 4 spaces.
        elif tkn_id == 25:
            end_pos = start_pos + 1  # Specific one-character token.
        elif tkn_id in [151, 152, 153, 154]:
            end_pos = start_pos + 6  # Multi-character tokens.
        elif tkn_id == 222:
            end_pos = start_pos + 3  # Special token case.
        else:
            end_pos = start_pos + (end_pos_original - start_pos_original) + space_between_tokens

        prev_end_pos = end_pos
        tkn_original_positions.append((satir_no, start_pos, end_pos))

    return tkn_named_entity, tkn_ids, tkn_original_positions, satir_no, start_pos, end_pos

import ast
    
def check_syntax(file_path):
    """
    Checks the syntax of a Python file.
    """
    with open(file_path, "r") as file:
        code = file.read()
    
        try:
            ast.parse(code)  # Parse to detect syntax errors.
            return None  
        except (SyntaxError, NameError) as e:
            line_number = e.lineno
            column_offset = e.offset
            error_message = e.msg
            return line_number, column_offset
        
def remove_text_from_file(file_path, line_number, start_column, end_column):
    """
    Removes text from a specific line and column range in a file.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        line_idx = line_number - 1
        
        if line_idx < 0 or line_idx >= len(lines):
            print(f"Warning: Line number {line_number} is invalid. The file has {len(lines)} lines.")
            return
        
        target_line = lines[line_idx]
        max_column = len(target_line)
        end_column = min(end_column, max_column)
        start_column = min(start_column, end_column)
        
        updated_line = target_line[:start_column] + target_line[end_column:]
        lines[line_idx] = updated_line
        
        with open(file_path, 'w') as file:
            file.writelines(lines)
            
    except Exception as e:
        print(f"File processing error: {e}")
        
def insert_text_to_file(file_path, line_num, start_col, end_col, text_to_insert):
    """
    Inserts text into a specific position in a file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        line_idx = line_num - 1
        
        if line_idx < 0 or line_idx >= len(lines):
            print(f"Warning: Line number {line_num} is invalid. The file has {len(lines)} lines.")
            return
            
        line = lines[line_idx]
        max_column = len(line)
        end_col = min(end_col, max_column)
        start_col = min(start_col, end_col)
        
        modified_line = line[:start_col] + text_to_insert + line[end_col:]
        lines[line_idx] = modified_line
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)
            
    except Exception as e:
        print(f"File processing error: {e}")
        
def insert_text_to_filev2(file_path, line_num, start_col, end_col, text_to_insert):
    """
    Inserts text into a specified line and column range within a file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        if line_num > len(lines) or line_num < 1:
            print("Invalid line number.")
            return

        line = lines[line_num-1] 
        modified_line = line[:end_col] + text_to_insert + line[end_col:]
        
        lines[line_num-1] = modified_line

        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)

    except FileNotFoundError:
        print(f"File {file_path} not found.")


import re
import os
import numpy as np
from numpy import array
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Attention, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy

def apply_suggestion_to_line(file_path, error_message):
    """
    Extracts line number and correction suggestion from an error message
    and applies the correction to the specified line in the given file.
    """
    line_number_match = re.search(r"line (\d+)", error_message)
    suggestion_match = re.search(r"Did you mean (.*?)(\))?\?", error_message)

    if line_number_match and suggestion_match:
        line_number = int(line_number_match.group(1))
        suggestion = suggestion_match.group(1)

        with open(file_path, 'r') as file:
            lines = file.readlines()

        lines[line_number] = suggestion + '\n'

        with open(file_path, 'w') as file:
            file.writelines(lines)

        print(f"Line {line_number} fixed: {suggestion}")
    else:
        print("Line number or correction suggestion not found.")

def split_sequence(sequence, n_steps_before, n_steps_after):
    """
    Splits a sequence into input-output pairs considering past and future steps.
    """
    X, y = list(), list()
    
    for i in range(len(sequence)):
        start_ix = max(0, i - n_steps_before)
        end_ix = min(len(sequence), i + 1 + n_steps_after)
        
        seq_x_before = sequence[start_ix:i]
        seq_x_after = sequence[i+1:end_ix]
        seq_x = np.concatenate((seq_x_before, seq_x_after), axis=0)
        seq_y = sequence[i]
        
        while len(seq_x) < n_steps_before + n_steps_after:
            seq_x = np.insert(seq_x, 0, -1)  # Padding if necessary
        
        X.append(seq_x)
        y.append(seq_y)
    
    return array(X), array(y)

import os
import re
import subprocess

# List to store corrected code snippets
duzeltilen_kodlar = []

# Counter for successfully processed codes
basarili_kod = 0

# Folder containing Python files
folder_path2 = 'indentation_error_data-ibm'

for filename2 in os.listdir(folder_path2):
    if filename2.endswith('.py'):  # Process only Python files
        file_path = os.path.join(folder_path2, filename2)
        output_file_name = file_path.replace('.py', '.txt')
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
            orjinal_kod = ''.join(lines)
            errorcode = ''.join(lines)
            
        # Fix Turkish character encoding issues
        errorcode = re.sub(r'Ä±', 'i', errorcode)
        errorcode = re.sub(r'Ã§', 'c', errorcode)
        errorcode = re.sub(r'Ä°', 'I', errorcode)
        errorcode = re.sub(r'Ã¼', 'u', errorcode)
        errorcode = re.sub(r'ÅŸ', 's', errorcode)
        errorcode = re.sub(r'ÄŸ', 'g', errorcode)
        errorcode = re.sub(r'Ã¶', 'o', errorcode)
        errorcode = re.sub(r'Ãœ', 'u', errorcode)
        errorcode = re.sub(r'\'', '"', errorcode)
        
        # Various regex-based text replacements for preprocessing
        errorcode = re.sub(r'\("[w\d\s+]"\n', ' LEFT_PARENTHESIS DOUBLE_QUOTE STRING DOUBLE_QUOTE NEWLINE ', errorcode)
        errorcode = re.sub(r'\{"[\w\d\s_-]+"', ' LEFT_BRACKET DOUBLE_QUOTE STRING DOUBLE_QUOTE ', errorcode)
        errorcode = re.sub(r'"[\w\d\s_-]+"', ' DOUBLE_QUOTE STRING DOUBLE_QUOTE ', errorcode)
        errorcode = re.sub(r'print\(".+"[\,\+\*\-]*.+\)', ' print LEFT_PARENTHESIS DOUBLE_QUOTE STRING DOUBLE_QUOTE5 COMMA var RIGHT_PARENTHESIS ', errorcode)
        
        # Additional regex replacements
        errorcode = re.sub(r'\'', ' DOUBLE_QUOTE3 \' ', errorcode)
        errorcode = re.sub(r'"', ' DOUBLE_QUOTE3  ', errorcode)
        errorcode = re.sub(r'\![^\=]', ' EXCLAMATION ', errorcode)
        errorcode = re.sub(r'    ', ' TABSPACE0 ', errorcode)
        errorcode = re.sub(r'\n', ' NEWLINE0 ', errorcode)
        
        # Tokenization processing
        deneme = call_moretokenizer(errorcode)
        id_sonuc = str(deneme[1])
        id_sonuc = re.sub(r'\[', '', id_sonuc)
        id_sonuc = re.sub(r'\,', '', id_sonuc)
        id_sonuc = re.sub(r'\]', '', id_sonuc)
        
        # Save the processed output
        with open(output_file_name, 'w') as output_file:
            output_file.write(id_sonuc)
        
        # Named entity recognition processing
        named_entity, ids, positions, satir_no, start_pos, end_pos = call_moretokenizer2(errorcode)
        
        # Run external Python type checker (Pyright)
        cmd = [
            os.path.join("win", "pyright-win.exe"),  # Platform bağımsız hale getirme
            "-t",
            "typeshed-fallback",
            file_path,
            "--outputjson"
            ]
        
        # Execute the command
        completed_process = subprocess.run(cmd, text=True, capture_output=True)
        
        # Save Pyright output
        with open('sonuc_ibm.txt', 'w') as file:
            file.write(completed_process.stdout)
            
        import json

        # Read JSON data from file
        with open('sonuc_ibm.txt', 'r') as file:
            data = json.load(file)

        # Retrieve the first error message
        first_error = data['generalDiagnostics'][1]

        # Extract start and end positions
        start_line = first_error['range']['start']['line'] + 1
        start_character = first_error['range']['start']['character']
        end_line = first_error['range']['end']['line']
        end_character = first_error['range']['end']['character']

        # Define the error range tuple
        error_range = (
            start_line,
            start_character,
            end_character,
        )

        # Check syntax of the file
        result = check_syntax(file_path)

        if result is None:
            indis = -1
            index_list = []
            for index, (satir, baslangic, bitis) in enumerate(positions):
                if satir == error_range[0] and (baslangic <= error_range[1]) and (bitis + 1 >= error_range[2]):
                    indis = index
        else:
            line_number, column_offset = result
            indis = -1
            index_list = []
            for index, (satir, baslangic, bitis) in enumerate(positions):
                if satir == line_number and (baslangic < column_offset) and (column_offset <= bitis):
                    indis = index
                elif satir == line_number and (baslangic < column_offset) and (column_offset > bitis):
                    indis = index

        import os
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        # Load training data
        texts = []
        labels = []
        folder_paths = ["missing_token_codes", "extra_token_codes", "wrong_token_codes"]

        # Read training data from specified folders
        for folder_path in folder_paths:
            folder_label = folder_paths.index(folder_path)
            for filename in os.listdir(folder_path):
                current_file_path = os.path.join(folder_path, filename)
                with open(current_file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    texts.append(text)
                    labels.append(folder_label)

        # Read test data
        test_texts = []
        test_labels = []
        with open(output_file_name, 'r', encoding='utf-8') as file:
            text = file.read()
            test_texts.append(text)

        # Tokenization and sequence padding
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        test_sequences = tokenizer.texts_to_sequences(test_texts)
        max_sequence_length = max([len(seq) for seq in tokenizer.texts_to_sequences(texts)])
        padded_test_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length)

        # Load trained model
        tur_model = tf.keras.models.load_model('correct_type_model')

        # Make predictions
        predictions = tur_model.predict(padded_test_sequences)
        predicted_labels = np.argmax(predictions, axis=1)

        # Print predictions
        for i, text in enumerate(test_texts):
            print(f"Text: {text} - Prediction: {predicted_labels[i]}")
        print(predicted_labels)

        # Read and preprocess test data for sequence prediction
        kelime = []
        with open(output_file_name, 'r') as f:
            temp = [line.strip() for line in f.readlines()]
            result = [word.split(' ') for word in temp]
            flat_list = [int(item) for sublist in result for item in sublist]
            kelime.extend(flat_list)

        kelime = np.array(kelime)

        n_steps_before = 6
        n_steps_after = 5
        X, y = split_sequence(kelime, n_steps_before, n_steps_after)

        # Reshape data for model input
        X = X.reshape((X.shape[0], X.shape[1], 1))
        X = X.astype(np.float64)
        y = y.astype(np.float64)

        # Load token prediction model
        id_model = tf.keras.models.load_model('token_predict_model')
        
        import shutil

backup_file_path = "duzeltilmisdosya.py"
shutil.copyfile(file_path, backup_file_path)

indis_limit = indis - len(ids)

while indis > indis_limit:
    success = False  # Flag to track success status

    for offset in [0, -1, 1]:  # Iterate over current index, previous, and next index
        if success:
            break

        try:
            input_data = X[indis].reshape(1, -1, 1)
            single_prediction = id_model.predict(input_data).flatten()

            # Sort probabilities and get top 5 predictions
            sorted_indices = single_prediction.argsort()[::-1]
            tahmin_listesi = sorted_indices[:5]

            for tahmin in tahmin_listesi:
                temizlenmis = positions[indis + offset]
                
                try:
                    if predicted_labels == 0:
                        insert_text_to_filev2(file_path, temizlenmis[0], temizlenmis[1], temizlenmis[2], listToPrint[tahmin])
                    elif predicted_labels == 2:
                        insert_text_to_file(file_path, temizlenmis[0], temizlenmis[1], temizlenmis[2], listToPrint[tahmin])
                    elif predicted_labels == 1:
                        remove_text_from_file(file_path, temizlenmis[0], temizlenmis[1], temizlenmis[2])
                except Exception as e:
                    print(f"Error occurred during processing: {e}")
                    continue
                
                try:
                    if predicted_labels == 0:
                        np.insert(X, indis, 222)
                        for adjustment in [-1, 1, 0]:  # Sequential attempts
                            try:
                                insert_text_to_filev2(file_path, temizlenmis[0], temizlenmis[1], temizlenmis[2]+adjustment, listToPrint[tahmin])
                                
                                with open(file_path, 'r') as file:
                                    code_to_compile = file.read()
                                
                                compile(code_to_compile, '<string>', 'exec')
                                success = True
                                break
                            except SyntaxError as e:
                                error_message = str(e)
                                suggestion_match = re.search(r" Missing parentheses in call to 'print'. Did you mean (.*?)(\))?\?", error_message)

                                if suggestion_match:
                                    suggestion = suggestion_match.group(1)
                                    print(f"Applying suggested fix: {suggestion}")
                                    apply_suggestion_to_line(file_path, error_message)
                                    
                                    with open(file_path, 'r') as file:
                                        code_to_compile = file.read()
                                    
                                    compile(code_to_compile, '<string>', 'exec')
                                    print("Error fixed and code is running.")
                                    success = True
                                    break
                                else:
                                    shutil.copyfile(backup_file_path, file_path)
                                    continue

                    elif predicted_labels == 2:
                        insert_text_to_file(file_path, temizlenmis[0], temizlenmis[1], temizlenmis[2], listToPrint[tahmin])
                    elif predicted_labels == 1:
                        remove_text_from_file(file_path, temizlenmis[0], temizlenmis[1], temizlenmis[2])

                    with open(file_path, 'r') as file:
                        code_to_compile = file.read()

                    compile(code_to_compile, '<string>', 'exec')
                    print("Error resolved")
                    basarili_kod += 1
                    duzeltilen_kodlar.append(file_path)
                    success = True
                    break

                except (SyntaxError, NameError, TypeError, AttributeError) as e:
                    shutil.copyfile(backup_file_path, file_path)
                    continue

                if success:
                    break

        except IndexError:  # If index exceeds positions list boundaries
            print("Error could not be resolved.")
            continue

    if not success:  # If no prediction was successful, decrement index
        indis -= 1
        print("Error could not be resolved.")
        continue
    else:
        break  # Exit loop if success is achieved

duzeltilenkodlar_str = '\n'.join(duzeltilen_kodlar)
dosya_adi = "sonuc-breaked_code_girinti.txt"
dosya_modu = "w"  # "w" mode overwrites existing file

# Write results to file
with open(dosya_adi, dosya_modu) as dosya:
    dosya.write(str(basarili_kod))  # Convert int to string before writing
    dosya.write(duzeltilenkodlar_str)
    dosya.write("\n")

