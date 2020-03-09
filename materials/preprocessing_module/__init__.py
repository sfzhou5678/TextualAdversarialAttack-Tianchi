import re
import mafan

REGEX_TO_REMOVE = re.compile(r"[^\u4E00-\u9FA5a-zA-Z0-9\!\@\#\$\%\^\&\*\(\)\-\_\+\=\`\~\\\|\[\]\{\}\:\;\"\'\,\<\.\>\/\?\ \t，。！？]")

def preprocess_text(text, truncate_at=100):
    '''
    清除限定字符集外的内容，并截取前若干字符的文本.
    '''
    truncated = text[:truncate_at]
    cleaned = REGEX_TO_REMOVE.sub(r'', truncated)

    return mafan.simplify(cleaned)