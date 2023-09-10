import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from datasets import Dataset
import pickle

class CustomBert:

    def __init__(self, model_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.label_encoder = pickle.load(open(model_path + '/LabelEncoder.pkl', 'rb'))
        self.chunk_size = 6 * 512

    def chunks_gen(self, string):
        """Produce `n`-character chunks from `s`."""
        i = 0
        for start in range(0, len(string), self.chunk_size):
            i += 1
            yield i, string[start:start + self.chunk_size]

    def split_text_on_parts(self, text):
        parts = []
        for part_number, chunk in self.chunks_gen(text):
            part = {'number': part_number, 'text': chunk}
            parts.append(part)
        df = pd.DataFrame(parts)
        return df

    def label_to_rating(self, label):
        rang_score = 0.058
        replace_dict = {
            'AAA': 16 * rang_score,
            'AA+': 15 * rang_score,
            'AA': 14 * rang_score,
            'AA-': 13 * rang_score,
            'A+': 12 * rang_score,
            'A': 11 * rang_score,
            'A-': 10 * rang_score,
            'BBB+': 9 * rang_score,
            'BBB': 8 * rang_score,
            'BBB-': 7 * rang_score,
            'BB+': 6 * rang_score,
            'BB': 5 * rang_score,
            'BB-': 4 * rang_score,
            'B+': 3 * rang_score,
            'B': 2 * rang_score,
            'B-': 1 * rang_score,
            'C': 0 * rang_score,
        }
        return replace_dict[label]
    def get_tokens(self, text):
        result = self.tokenizer(text.to_list(), padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        return result
    def predict(self, text):
        parts = self.split_text_on_parts(text)
        inputs = self.get_tokens(parts['text'])
        output = self.model.forward(**inputs)
        predicts = output['logits']
        predicts_index = torch.argmax(predicts, axis=1).numpy()
        parts['labels'] = self.label_encoder.inverse_transform(predicts_index)
        parts['rating'] = parts['labels'].apply(self.label_to_rating)
        predict_label = parts['labels'].iloc[0]
        return predict_label

if __name__ == "__main__":
    test = pd.read_csv('/private/var/hosting/rech-recognition/tests/hackaton/test_cuted_version.csv')
    model = CustomBert('/private/var/hosting/rech-recognition/models/AKRA_best') # указать путь до нужной модели
    test['prediction'] = test['text'].apply(model.predict) # model.predict(text) если нужно для одлного текста
    f1_score = f1_score(test['lvl'], test['prediction'], average='weighted')
    pass
