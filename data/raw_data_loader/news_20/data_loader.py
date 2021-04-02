import os
import re


from data_preprocessing.base.base_raw_data_loader import TextClassificationRawDataLoader

_QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                           r'|^In article|^Quoted from|^\||^>)')


class RawDataLoader(TextClassificationRawDataLoader):
    def __init__(self, data_path):
        super().__init__(data_path)
        self.train_path = "20news-bydate-train"
        self.test_path = "20news-bydate-test"

    def load_data(self):
        if len(self.X) == 0 or len(self.Y) == 0 or self.attributes["label_vocab"] is None:
            train_size = 0
            for root1, dirs, _ in os.walk(os.path.join(self.data_path, self.train_path)):
                for d in dirs:
                    for root2, _, files in os.walk(os.path.join(root1, d)):
                        for file_name in files:
                            file_path = os.path.join(root2, file_name)
                            self.process_data_file(file_path)
                            self.Y[train_size] = d
                            train_size += 1
            test_size = 0
            for root1, dirs, _ in os.walk(os.path.join(self.data_path, self.test_path)):
                for d in dirs:
                    for root2, _, files in os.walk(os.path.join(root1, d)):
                        for file_name in files:
                            file_path = os.path.join(root2, file_name)
                            self.process_data_file(file_path)
                            self.Y[train_size+test_size] = d
                            test_size += 1
            self.attributes["train_index_list"] = [i for i in range(train_size)]
            self.attributes["test_index_list"] = [i for i in range(train_size, train_size+test_size)]
            self.attributes["index_list"] = self.attributes["train_index_list"] + self.attributes["test_index_list"]
            self.attributes["label_vocab"] = {label: i for i, label in enumerate(set(self.Y.values()))}

    # remove header
    def remove_headers(self, text):
        _before, _blankline, after = text.partition('\n\n')
        return after

    # remove quotes
    def remove_quotes(self, text):
        good_lines = [line for line in text.split('\n')
                      if not _QUOTE_RE.search(line)]
        return '\n'.join(good_lines)

    # remove footers
    def remove_footers(self, text):
        lines = text.strip().split('\n')
        for line_num in range(len(lines) - 1, -1, -1):
            line = lines[line_num]
            if line.strip().strip('-') == '':
                break

        if line_num > 0:
            return '\n'.join(lines[:line_num])
        else:
            return text

    def process_data_file(self, file_path):
        cnt = 0
        with open(file_path, "r", errors='ignore') as f:
            content = f.read()
            content = content.replace("\n", " ")
            # content = self.remove_headers(content)
            # content = self.remove_footers(content)
            # content = self.remove_quotes(content)
            idx = len(self.X)
            self.X[idx] = content
            cnt += 1
        return cnt


class ClientDataLoader(BaseClientDataLoader):

    def __init__(self, data_path, partition_path, client_idx=None, partition_method="uniform", tokenize=False):
        data_fields = ["X", "Y"]
        super().__init__(data_path, partition_path, client_idx, partition_method, tokenize, data_fields)
        self.clean_data()
        if self.tokenize:
            self.tokenize_data()

    def tokenize_data(self):
        def __tokenize_data(data):
            for i in range(len(data["X"])):
                data["X"][i] = data["X"][i].split(" ")

        __tokenize_data(self.train_data)
        __tokenize_data(self.test_data)

    def clean_data(self):
        def __clean_data(data):
            for i in range(len(data["X"])):
                data["X"][i] = self.clean_str(data["X"][i])
        __clean_data(self.train_data)
        __clean_data(self.test_data)

    def clean_str(self, string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()