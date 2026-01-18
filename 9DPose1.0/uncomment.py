import os
import tokenize
from io import BytesIO

def remove_comments(code):
    tokens = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
    result = []
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            result.append(tok)
    return tokenize.untokenize(result).decode('utf-8')

if __name__ == "__main__":
    for root, _, files in os.walk("./"):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as fr:
                    code = fr.read()
                new_code = remove_comments(code)
                with open(path, "w", encoding="utf-8") as fw:
                    fw.write(new_code)
