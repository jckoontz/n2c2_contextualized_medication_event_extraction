import pandas as pd
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sentence_getter import SentenceGetter
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from seqeval.metrics import accuracy_score,f1_score



def get_enumerated_sentences(file:str) -> pd.DataFrame:
    with open(file) as file:
      lines = [line.split() for line in file]
    enumerated_sentences = []
    sentence_number = 1
    for i in range(len(lines)):
        if not lines[i]:
            sentence_number += 1
        else:
            row = [sentence_number, lines[i][-1], lines[i][0]]
            if row[1] == "#":
              pass
            else:
              enumerated_sentences.append(row)
    enumerated_dataframe = pd.DataFrame(enumerated_sentences, columns=['Sentence #', 'Word', 'Tag'])
    enumerated_dataframe.loc[:,'Word'] = enumerated_dataframe['Word'].astype(str)
    return enumerated_dataframe


def replace_labels(dataframe: pd.DataFrame) -> pd.DataFrame:
  dataframe.loc[:,'Word'] = dataframe['Word'].astype(str)
  dataframe['Tag'] = dataframe['Tag'].replace('(No)?Disposition','Medication', regex=True)
  dataframe['Tag'] = dataframe['Tag'].replace('Undetermined','Medication', regex=True)
  return dataframe


def get_sentences_and_labels(getter: SentenceGetter) -> list:
    sentences = [[s[0] for s in sent] for sent in getter.sentences]
    labels = [[s[1] for s in sent] for sent in getter.sentences]
    return sentences, labels


def get_tag_vals(dataframe: pd.DataFrame):
    tags_vals = list(set(dataframe["Tag"].values))
    tags_vals.append('[CLS]')
    tags_vals.append('[SEP]')
    return tags_vals


def get_tag2idx(tags_vals: list) -> dict:
  return {t: i for i, t in enumerate(tags_vals)}


def get_tag2name(tag2idx: dict) -> dict:
  return {tag2idx[key] : key for key in tag2idx.keys()}


def bert_tokenize(sentences: list, labels: list, tokenizer) -> tuple:
    tokenized_texts = []
    word_piece_labels = []
    for sentence, label in (zip(sentences,labels)):
        temp_label = []
        temp_token = []
        # Add [CLS] at the front 
        temp_label.append('[CLS]')
        temp_token.append('[CLS]')
        for word, label in zip(sentence, label):
          token_list = tokenizer.tokenize(word)
          temp_token.extend(token_list) 
          n_subwords = len(token_list)    
          temp_label.extend([label] * n_subwords)
        # Add [SEP] at the end
        temp_label.append('[SEP]')
        temp_token.append('[SEP]')
        tokenized_texts.append(temp_token)
        word_piece_labels.append(temp_label)
    return tokenized_texts, word_piece_labels


def get_input_id_tensors(train_inputs: list, val_inputs: list, test_inputs: list) -> tuple:
    train_inputs = torch.tensor(train_inputs)
    val_inputs = torch.tensor(val_inputs)
    test_inputs = torch.tensor(test_inputs)
    return train_inputs, val_inputs, test_inputs


def get_tag_tensors(train_tags: list, val_tags: list, test_tags: list) -> tuple:
    train_tags = torch.tensor(train_tags)
    val_tags = torch.tensor(val_tags)
    test_tags = torch.tensor(test_tags)
    return train_tags, val_tags, test_tags


def get_mask_tensors(train_masks: list, val_masks: list, test_masks: list) -> tuple:
    train_masks = torch.tensor(train_masks)
    val_masks = torch.tensor(val_masks)
    test_masks = torch.tensor(torch.as_tensor(test_masks))
    return train_masks, val_masks, test_masks


def prepare_data_helper(config: dict) -> tuple:
    train = get_enumerated_sentences(config['train'])
    test = get_enumerated_sentences(config['dev']) 
    train = replace_labels(train)
    test = replace_labels(test)
    train_getter = SentenceGetter(train)
    test_getter = SentenceGetter(test)
    train_sentences, train_labels = get_sentences_and_labels(train_getter)
    test_sentences, test_labels = get_sentences_and_labels(test_getter)
    tags_vals = get_tag_vals(train)
    tag2idx = get_tag2idx(tags_vals)
    tag2name = get_tag2name(tag2idx)
    MAX_LEN  = config['MAX_LEN']

    tokenizer = AutoTokenizer.from_pretrained(config['pretrained_model'], do_lowercase=False)

    train_tokenized_texts, train_word_piece_labels = bert_tokenize(train_sentences, train_labels, tokenizer)
    test_tokenized_texts, test_word_piece_labels = bert_tokenize(test_sentences, test_labels, tokenizer)

    train_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in train_tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    test_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in test_tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    train_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in train_word_piece_labels],
                        maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                        dtype="long", truncating="post")
    test_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in test_word_piece_labels],
                        maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                        dtype="long", truncating="post")

    train_attention_masks = [[int(i>0) for i in ii] for ii in train_input_ids]
    test_attention_masks = [[int(i>0) for i in ii] for ii in test_input_ids]

    train_input, val_input, train_tags, val_tags = train_test_split(train_input_ids, train_tags,
                                                            random_state=2022, test_size=0.1)
    train_masks, val_masks, _, _ = train_test_split(train_attention_masks, train_input_ids,
                                             random_state=2022, test_size=0.1)
    
    train_inputs, val_inputs, test_inputs = get_input_id_tensors(train_input, val_input, test_input_ids)
    train_tags, val_tags, test_tags = get_tag_tensors(train_tags, val_tags, test_tags)

    train_masks, val_masks, test_masks = get_mask_tensors(train_masks, val_masks, test_attention_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_tags)
    val_data = TensorDataset(val_inputs, val_masks, val_tags)
    test_data = TensorDataset(test_inputs, test_masks, test_tags)

    return train_data, val_data, test_data, tags_vals, tag2name

def get_metrics(best_path: list, tag2name: dict, attention_masks, labels: list, loss: float) -> dict:
      pred, true = [], []
      for i in range(len(best_path)): 
        idx = len(torch.where(attention_masks[i] >= 1)[0].detach().cpu().numpy().tolist())
        true_ = labels[i][:idx].detach().cpu().numpy().tolist()
        pred.append([tag2name[token] for token in best_path[i][0][1:-1]])
        true.append([tag2name[token] for token in true_][1:-1])
      accuracy = accuracy_score(true, pred)
      f1 = f1_score(true, pred)
      return {'loss':loss, 'acc': accuracy, 'f1': f1}
    
