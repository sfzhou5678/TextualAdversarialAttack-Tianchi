"""
For Python 3.6+
开放资料 各组件 使用示例.
"""


def reference_model(model_path='reference_model/mini.ftz', test_args=('你好',)):
    """
    调用参考模型; 需要fasttext
    """
    import fasttext
    model = fasttext.load_model(model_path)
    print('----Reference model prediction----')
    print(*test_args, model.predict(*test_args))

def distance_measure(test_args=(['你好呀'], ['你好'])):
    """
    调用距离计算器; 需要gensim, numpy
    """
    from distance_module import DistanceCalculator
    dc = DistanceCalculator(EMBEDDING_PATH='../data/materials/zh.300.vec.gz')
    print('----Distance measure----')
    print(*test_args, dc(*test_args))

def preprocess_example(test_args=('慶曆四年春，んóμ,滕（Tteéng）子京謫（zhé）守巴陵郡。越明年，政通人和，百廢俱興。乃 重修岳陽樓，增其舊制，刻唐賢、今人詩賦於其上。',)):
    """
    调用评测中使用的预处理函数; 需要mafan
    """
    from preprocessing_module import preprocess_text
    print('----Text preprocessing----')
    # print(*test_args, preprocess_text(*test_args))
    print(preprocess_text(*test_args))


if __name__ == '__main__':
    # reference_model(model_path='../data/materials/mini.ftz',test_args=['一屋子的辣鸡'])
    # reference_model(model_path=r"C:\Users\zsf\Downloads\mini-explicit-labels.ftz",test_args=['一屋子的辣鸡'])

    # distance_measure()
    preprocess_example(('$hi,m@',))
