# *-* encoding: utf-8 *-*

import json
import os.path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix


def get_inds(inp_corpus):
    """Возвращает словарь индексов документов для каждого типа имитации.
    Parameters
    ----------
    inp_corpus: list
        Корпус в JSON формате

    Returns
    -------
    d_inds: dict
        Словарь с индексами слов, по типам имитации.
        a -- без имитации, "no_im",
        b -- имитация пола, "gender_im",
        v -- имитация стиля, "style_im"
    """

    d_inds = {}
    for doc_ind, doc in enumerate(inp_corpus):
        if doc["meta"]["imitation_type"] == "no_im":
            if "a" not in d_inds:
                d_inds["a"] = []
            d_inds["a"].append(doc_ind)
        elif doc["meta"]["imitation_type"] == "gender_im":
            if "b" not in d_inds:
                d_inds["b"] = []
            d_inds["b"].append(doc_ind)
        elif doc["meta"]["imitation_type"] == "style_im":
            if "v" not in d_inds:
                d_inds["v"] = []
            d_inds["v"].append(doc_ind)

    return d_inds


def balance_corpus(inp, y_meta_field="gender", shuffle_docs=False):
    """Балансирует корпус по метке y.
    Parameters
    ----------
    inp: list
        Корпус в формате JSON

    y_meta_field: str
        Название поля для балансировки, на основе значений меток из этого поля будет проводится балансировка.

    shuffle_docs: bool
        Флаг, перемешивать или нет документы.
        Если False берётся n первых документов, где n -- количество документов в самом маленьком классе.
        Если True, то берётся n случайных документов.

    Returns
    -------
    balanced_inp: list
        Сбалансированный корпус
    """

    d_docs_by_classes = {}
    for doc in inp:
        y_doc = doc["meta"][y_meta_field]
        if y_doc not in d_docs_by_classes:
            d_docs_by_classes[y_doc] = []
        d_docs_by_classes[y_doc].append(doc)

    min_len = min([len(v) for v in d_docs_by_classes.values()])

    res = []
    for k, v in d_docs_by_classes.items():
        res += v[:min_len]

    return res


def balance_data(X, Y, strategy="undersampling", shuffle_data=True, random_state=None):
    """
    Балансировка множества.

    Parameters
    ----------
    X: list
        Список примеров для балансировки

    Y: list
        Список меток для примеров, используется при балансировке

    strategy: str, {"undersampling", "oversampling"}
        undersampling -- отбрасываем последние примеры в наиболее представительных классах до размера наименее представитьельного;
        oversampling -- случайным образом копируем примеры в малопредставительных классах до размера наиболее представительного.

    shuffle_data: boolean
        Если True, то перед балансировкой производится перемешивание примеров внутри класса.

    random_state: None, int
        Используется при случайном перемешивании и копировании примеров в oversampling.
    """

    d_classes_sample_inds = {}
    for sample_ind, y in enumerate(Y):
        if y not in d_classes_sample_inds:
            d_classes_sample_inds[y] = []
        d_classes_sample_inds[y].append(sample_ind)

    np_random = np.random.RandomState(random_state)
    if shuffle_data:
        for k, v in d_classes_sample_inds.items():
            v = np.array(v)
            np_random.shuffle(v)
            d_classes_sample_inds[k] = v

    class_lenses = [len(v) for v in d_classes_sample_inds.values()]
    max_class_len = max(class_lenses)
    min_class_len = min(class_lenses)

    balanced_d_classes_sample_inds = {}
    for k, v in d_classes_sample_inds.items():
        if strategy == "undersampling":
            balanced_d_classes_sample_inds[k] = v[:min_class_len]
        elif strategy == "oversampling":
            if len(v) < max_class_len:
                balanced_d_classes_sample_inds[k] = np.concatenate(
                    [v, np.array(v)[np_random.randint(0, len(v), max_class_len - len(v))]], axis=0)
            else:
                balanced_d_classes_sample_inds[k] = v

    res_x = []
    res_y = []
    for k, v in balanced_d_classes_sample_inds.items():
        for val_ind in v:
            res_x.append(X[val_ind])
            res_y.append(Y[val_ind])
    return res_x, res_y


def filter_docs_by_length_in_words(inp, max_doc_len):
    if max_doc_len is not None:
        l_doc_lengths = []
        for doc in inp:
            doc_len_in_words = 0
            for sent in doc["sentences"]:
                doc_len_in_words += len(sent)
            l_doc_lengths.append(doc_len_in_words)
        res = [doc for doc, doc_len in zip(inp, l_doc_lengths) if doc_len <= max_doc_len]
    else:
        res = inp
    return res


def get_scores_dict(y_true, y_pred):
    """Оценить точность по правильным и предсказанным меткам.
    Parameters
    ----------
    y_true: list
        Правильные метки

    y_pred: list
        Предсказанные метки

    Returns
    -------
    res: dict
        Словарь с метриками:
        F1 -- F1 score, взвешенный
        acc -- accuracy
        classification_report -- classification_report
        confusion_matrix -- confusion_matrix
    """
    res = {"F1": f1_score(y_true, y_pred, average="weighted"),
           "acc": accuracy_score(y_true, y_pred),
           "classification_report": classification_report(y_true, y_pred, digits=3),
           "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()}
    return res


def f_eval_model(y_true, y_pred, d_inds=None):
    """Функция рассчитывает меткрики оценки для каждого типа имитации (если заданы d_inds).
    Parameters
    ----------
    y_true: list
        Правильные метки

    y_pred: list
        Предсказанные метки

    d_inds: dict
        Словарь с индексами меток

    Returns
    -------
    res: dict
        Словарь с метриками из get_scores_dict для каждого типа имитации
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    res = {"all": get_scores_dict(y_true, y_pred),
           "a": {}, "b": {}, "v": {}}

    if d_inds is not None:
        for k, v in d_inds.items():
            try:
                res[k] = get_scores_dict(y_true[v], y_pred[v])
            except:
                res[k] = None

    return res


def balance_by_user_ids_by_gender(inp, shuffle_users=False):
    """Балансирует корпус по пользователям"""

    l_user_id_male, l_user_id_female = set(), set()
    for doc_user_id_by_ind, doc in enumerate(inp):
        doc_gender = int(doc["meta"]["gender"])
        doc_user_id = doc["meta"].get("user_id", doc_user_id_by_ind)
        if doc_gender == 0:
            l_user_id_female.add(doc_user_id)
        else:
            l_user_id_male.add(doc_user_id)
    l_user_id_female = list(l_user_id_female)
    l_user_id_male = list(l_user_id_male)

    if shuffle_users:
        np.random.shuffle(l_user_id_female)
        np.random.shuffle(l_user_id_male)

    min_len = min([len(l_user_id_female), len(l_user_id_male)])
    res_user_ids = l_user_id_female[:min_len] + l_user_id_male[:min_len]

    res = []
    for doc in inp:
        doc_user_id = doc["meta"]["user_id"]
        if doc_user_id in res_user_ids:
            res.append(doc)

    return res


def get_user_ids(inp):
    d_user_ids = {}
    for doc in inp:
        doc_user_id = doc["meta"]["user_id"]
        doc_gender = int(doc["meta"]["gender"])
        d_user_ids[doc_user_id] = doc_gender

    l_user_ids = []
    l_user_ids_gender = []
    for k, v in d_user_ids.items():
        l_user_ids.append(k)
        l_user_ids_gender.append(v)

    l_user_ids = np.array(l_user_ids)
    l_user_ids_gender = np.array(l_user_ids_gender)

    return l_user_ids, l_user_ids_gender


def filter_by_user_id(inp, user_ids):
    return np.array([doc for doc in inp if doc["meta"]["user_id"] in user_ids])


def get_y_gender(inp):
    return np.array([int(doc["meta"]["gender"]) for doc in inp])


def get_y(inp, meta_key):
    """
    Возвращает список занчений ключа meta_key для документов из inp.

    Parameters
    ----------
    inp: list
        Корпус в формате JSON

    meta_key: str
        Ключ в поле "meta"

    Returns
    -------
    res: np.array
        Список значений ключа meta_key для документов из inp

    """
    return np.array([doc["meta"][meta_key] for doc in inp])


def flatten_d_eval(d_eval, **args):
    d_res = {"F1 all": d_eval["all"]["F1"],
             "F1 a": d_eval["a"].get("F1", ""),
             "F1 b": d_eval["b"].get("F1", ""),
             "F1 v": d_eval["v"].get("F1", "")}
    for k, v in args.items():
        d_res[k] = v

    return d_res


def get_texts(inp):
    res = []
    for doc in inp:
        res.append(doc["text"])
    return np.array(res)


def flatten_d_eval(d_eval, **args):
    d_res = {"F1 all": d_eval["all"]["F1"],
             "F1 a": d_eval["a"].get("F1", ""),
             "F1 b": d_eval["b"].get("F1", ""),
             "F1 v": d_eval["v"].get("F1", "")}
    for k, v in args.items():
        d_res[k] = v

    return d_res


class MyDataSet:

    def __init__(self, inp):
        self.y = np.array([int(doc["meta"]["gender"]) for doc in inp])
        self.texts = get_texts(inp)
        self.inp = inp
        self.d_inds = get_inds(inp)


def eval_model(y_true, y_pred, d_inds, d_inds_balance):
    res = {"all_a": f1_score(y_true[d_inds["a"]], y_pred[d_inds["a"]], average="weighted", pos_label=None),
           "all_b": f1_score(y_true[d_inds["b"]], y_pred[d_inds["b"]], average="weighted", pos_label=None),
           "all_v": f1_score(y_true[d_inds["v"]], y_pred[d_inds["v"]], average="weighted", pos_label=None),
           "balance_a": f1_score(y_true[d_inds_balance["a"]], y_pred[d_inds_balance["a"]], average="weighted", pos_label=None),
           "balance_b": f1_score(y_true[d_inds_balance["b"]], y_pred[d_inds_balance["b"]], average="weighted", pos_label=None),
           "balance_v": f1_score(y_true[d_inds_balance["v"]], y_pred[d_inds_balance["v"]], average="weighted", pos_label=None)}
    return res


def split_by_user_id(inp, train_size=0.9, test_size=None, random_state=None, y=None):
    if test_size is None:
        test_size = 1.0 - train_size

    user_ids = set()
    user_id_ind = 0
    for doc in inp:
        doc_user = doc["meta"].get("user_id", user_id_ind)
        if "user_id" not in doc["meta"]:
            user_id_ind += 1
        user_ids.add(doc_user)

    user_ids = list(user_ids)

    train_user_id, test_user_id = train_test_split(user_ids,
                                                   train_size=train_size, test_size=test_size,
                                                   random_state=random_state)

    res_train = []
    res_valid = []
    user_id_ind = 0
    for doc_ind, doc in enumerate(inp):
        doc_user = doc["meta"].get("user_id", user_id_ind)
        if "user_id" not in doc["meta"]:
            user_id_ind += 1
        if doc_user in train_user_id:
            res_train.append(doc)
        else:
            res_valid.append(doc)

    if y is not None:
        res_train_y = []
        res_valid_y = []
        user_id_ind = 0
        for doc_ind, doc in enumerate(inp):
            doc_user = doc["meta"].get("user_id", user_id_ind)
            if "user_id" not in doc["meta"]:
                user_id_ind += 1
            if doc_user in train_user_id:
                res_train_y.append(y[doc_ind])
            else:
                res_valid_y.append(y[doc_ind])

    if y is None:
        return res_train, res_valid
    else:
        return res_train, res_valid, res_train_y, res_valid_y


def filter_a_b_by_user_id(inp):
    """
    Фильтрация документов, оставляет документы только типа а -- без имитации и б -- имитация пола.
    При этом для каждого типа документов свой уникальных наор пользователей.
    Т.е. получается, что тексты а пишут одни пользователи, б пишут другие пользователи.

    Parameters
    ----------
    inp: list
        Список документов в формате JSON

    Returns
    -------
    filtered_inp: list
        Список отфильтрованных документв
    """

    set_user_ids = set()
    for doc in inp:
        if doc["meta"]["imitation_type"] in ["no_im", "gender_im"]:
            set_user_ids.add(doc["meta"]["user_id"])

    a_user_ids, b_user_ids = train_test_split(list(set_user_ids), train_size=0.5, test_size=0.5)

    res_a = []
    res_b = []
    for doc in inp:
        doc_type = doc["meta"]["imitation_type"]
        doc_user_id = doc["meta"]["user_id"]

        if doc_type == "no_im" and doc_user_id in a_user_ids:
            res_a.append(doc)
        elif doc_type == "gender_im" and doc_user_id in b_user_ids:
            res_b.append(doc)

    return res_a + res_b


def filter_by_imitation_type(inp, l_types):
    """
    Фильтрация документов, оставляем только документы с типом имитации из l_types.

    Parameters
    ----------
    inp: list
        Список документов в формате JSON.

    l_types: list
        Список разрещенных типов имитации.
        Возможные значения (одно или нескольок): no_im, gender_im, style_im.

    Returns
    -------
    filtered_inp: list
        Список отфильтрованных документв
    """

    res = []
    for doc in inp:
        doc_imitation_type = doc["meta"]["imitation_type"]
        if doc_imitation_type in l_types:
            res.append(doc)
    return res


def filter_by_percentile_docs_lens(inp):
    # Удаляем слишком коротки и слишком длинные документы
    docs_len = []
    for doc in inp:
        doc_len = sum([len(sent) for sent in doc["sentences"]])
        docs_len.append(doc_len)
    docs_len = np.array(docs_len)

    min_len, max_len = np.percentile(docs_len, [5, 95])
    norm_inds = np.arange(0, len(inp))[(min_len < docs_len)&(docs_len < max_len)]
    res = np.array(inp)[norm_inds]
    return res


def load_corpus_from_config(data_set):
    # Считываем корпус
    f = open(os.path.expandvars(data_set["inp"]))
    inp = json.load(f)
    f.close()

    # Проверяем, надо ли проводить фильтрацию по типам имитации
    filter_imitation_types = data_set.get("filter", [])
    if len(filter_imitation_types) > 0:
        if "filter_ab_by_users" in filter_imitation_types:
            inp = filter_a_b_by_user_id(inp)
        else:
            inp = filter_by_imitation_type(inp, filter_imitation_types)

    # Проверяем, надо ли делить на тренировочное и валидационное множество
    if data_set.get("take_valid", 0.) > 0:
        take_valid_by = data_set.get("take_valid_by", "users")
        if take_valid_by == "users":
            inp_train, inp_valid = split_by_user_id(inp, 1.0 - data_set["take_valid"])
        else:
            inp_train, inp_valid = train_test_split(inp, train_size=1.0 - data_set["take_valid"],
                                                    test_size=data_set["take_valid"])
    else:
        inp_train = inp
        inp_valid = []

    # Балансировка корпуса по документам для каждого класса
    if data_set.get("balance", False):
        inp_train = balance_corpus(inp_train, data_set["balance_by"])
        if len(inp_valid) > 0:
            inp_valid = balance_corpus(inp_valid, data_set["balance_by"])

    return inp_train, inp_valid

def load_corpus(inp):
    """
    Загрузка корпуса

    Parameters
    ----------
    inp: {str;list}
        Если строка -- то это имя файла с корпусом в JSON формате
        Если список -- передаёт на выход в неизменном виде

    Returns
    -------
    res: list
        Загруженный корпус
    """

    if type(inp) in [str, unicode]:
        f = open(os.path.expandvars(inp), "r")
        res = json.load(f)
        f.close()
    else:
        res = inp
    return res


def combine_docs(list_docs):
    """
    Объединяет список документов в один документв.
    В качестве мета информации берётся метаинформация первого документа.

    Parameters
    ----------
    list_docs: list
        Список документов в формате JSON

    Returns
    -------
    combine_document: json
        Объединённый документ в формате JSON
    """

    #TODO добавить обновление позиций начала слов в предложениях после объединения
    res_meta = list_docs[0]["meta"]
    res_text = "".join([doc["text"] for doc in list_docs])
    res_sentences = []
    for doc in list_docs:
        res_sentences += doc["sentences"]
    return {"meta": res_meta,
            "text": res_text,
            "sentences": res_sentences}


def combine_by_author_id(inp):
    """
    Объединяет все документы от одного автора в один текст.
    Испольузет для объединеня функцию combine_docs.

    Parameters
    ----------
    inp: list
        Список документов в формате JSON

    Returns
    -------
    combined_documents: list
        Список объединённых документов.
    """
    
    res = {}
    for doc in inp:
        doc_user_id = doc["meta"]["user_id"]
        if doc_user_id not in res:
            res[doc_user_id] = []
        res[doc_user_id].append(doc)

    l_res = []
    for k, v in res.items():
        l_res.append(combine_docs(v))
    return l_res


if __name__ == "__main__":
    # import json
    # f = open("../small_data/gender_Imitation_Toloka_small.json", "r")
    # inp_small = json.load(f)
    # f.close()
    data_set = {"inp": "$SAG_DATASETS_DIR/Gender_new_format/gender_Imitation_Toloka.json",
                "take_valid": 0.5,
                "take_valid_by": "users",
                "balance": True,
                "balance_by": "gender",
                "filter": ["no_im", "gender_im"]}
    trn_data, val_data = load_corpus_from_config(data_set)
    print(len(trn_data))
    print(len(val_data))

    for doc_trn in trn_data:
        for doc_val in val_data:
            if doc_trn["text"] == doc_val["text"]:
                print(1)
            if doc_trn["meta"]["user_id"] == doc_val["meta"]["user_id"]:
                print(2)
    print("Выполнено")
