from conll_dictorizer import CoNLLDictorizer


def load_conll2009_pos():
    train_file = '../../../corpus/conll2009/en/CoNLL2009-ST-English-train-pos.txt'
    dev_file = '../../../corpus/conll2009/en/CoNLL2009-ST-English-development-pos.txt'
    test_file = '../../../corpus/conll2009/en/CoNLL2009-ST-test-words-pos.txt'
    test2_file = 'simple_pos_test.txt'

    column_names = ['id', 'form', 'lemma', 'plemma', 'pos', 'ppos']

    train_sentences = open(train_file).read().strip()
    dev_sentences = open(dev_file).read().strip()
    test_sentences = open(test_file).read().strip()
    test2_sentences = open(test2_file).read().strip()
    return train_sentences, dev_sentences, test_sentences, column_names


BASE_DIR = 'Datasets/'


def load_conll2003_en():
    train_file = BASE_DIR + 'eng.train'
    dev_file = BASE_DIR + 'eng.valid'
    test_file = BASE_DIR + 'eng.test'
    column_names = ['form', 'ppos', 'pchunk', 'ner']
    train_sentences = open(train_file).read().strip()
    dev_sentences = open(dev_file).read().strip()
    test_sentences = open(test_file).read().strip()
    return train_sentences, dev_sentences, test_sentences, column_names


def create_dicts():
    train_sentences, dev_sentences, test_sentences, column_names = load_conll2003_en()

    conll_dict = CoNLLDictorizer(column_names, col_sep=' +')
    train_dict = conll_dict.transform(train_sentences)
    dev_dict = conll_dict.transform(dev_sentences)
    test_dict = conll_dict.transform(test_sentences)


    return train_dict, dev_dict, test_dict
