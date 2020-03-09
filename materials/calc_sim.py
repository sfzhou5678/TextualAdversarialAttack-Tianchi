import fasttext
import jieba
from materials.preprocessing_module import preprocess_text

model_path = '../data/materials/mini.ftz'
model = fasttext.load_model(model_path)

is_obscenity_dict = {'__label__0': 0,
                     '__label__1': 1}

# file = '../data/obscenities.txt'
file = '../data/corpus.txt'

wf = open('corpus_output.txt', 'w', encoding='utf-8')
with open(file, encoding='utf-8') as f:
  for line in f:
    text = line.strip()
    text = preprocess_text(text)
    text = ' '.join(jieba.cut(text))
    (lbl,), (score,) = model.predict(text)
    if not is_obscenity_dict[lbl]:
      score = 1 - score
    wf.write('%s\t%.6f\n' % (text, score))
wf.close()
