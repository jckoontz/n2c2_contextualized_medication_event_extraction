import sys
import os
import argparse
import yaml
import glob
import os
from tqdm import tqdm
import spacy
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer


from model import NER_CRF


def main(args=None):

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    config = load_config(args.config)

    nlp = spacy.load("en_core_web_sm")

    evaluation_documents = get_evaluation_documents(args.eval_data)

    ner_model = NER_CRF.load_from_checkpoint(config['checkpoint'],
                                             hparams_file=config['hparams'])

    process_sentences(evaluation_documents, nlp, "ner_model", args.outpath)


def load_config(config_path: str) -> dict:
    '''
    Load configuration file
    '''
    with open(config_path) as f:
        config = yaml.safe_load(f)
        return config


def get_evaluation_documents(path: str) -> list:
    return [file for file in glob.glob(os.path.join(path, '*.txt'))]


def get_offset_to_sentences_dict(text, nlp_model) -> dict:
    doc = nlp_model(text)
    offset_to_sentences = {}
    for sent in doc.sents:
        for word in sent:
            offset_to_sentences[word.idx] = word.text

    sentences = []
    tmp = []
    offset_tmp = {}
    offset_sentence_map = {}
    i = 0
    j = 0
    max_len = 0
    for offset, token in offset_to_sentences.items():
        if token == ' ' or j >= 300:
            if tmp:
                sentences.append(tmp)
                if len(tmp) > max_len:
                    max_len = len(tmp)
                offset_sentence_map[i] = {
                    'sentence': tmp, 'offsets': offset_tmp}
                i += 1
                j = 0
                tmp = []
                offset_tmp = {}
        else:
            j += 1
            tmp.append(str(token))
            offset_tmp[offset] = str(token)
    return offset_sentence_map, sentences

def prepare_batch_for_inference(sentences: list):
    MAX_LEN  = 300
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", do_lowercase=False)
    tokenized_texts = bert_tokenize_for_inference(sentences, tokenizer)
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = [[int(i>0) for i in ii] for ii in input_ids]
    input_ids_tensors = torch.tensor(input_ids)
    attention_masks_tensors = torch.tensor(attention_masks)
    dataset = TensorDataset(input_ids_tensors, attention_masks_tensors)
    dataloader = DataLoader(dataset, batch_size=2)
    return dataloader, tokenized_texts


def bert_tokenize_for_inference(sentences: list, tokenizer) -> tuple:
    tokenized_texts = []
    for sentence in tqdm(sentences):
        temp_token = []
        # Add [CLS] at the front 
        temp_token.append('[CLS]')
        for word in sentence:
          token_list = tokenizer.tokenize(word)
          temp_token.extend(token_list) 
        # Add [SEP] at the end
        temp_token.append('[SEP]')
        tokenized_texts.append(temp_token)

    return tokenized_texts

def get_predictions(ner_model, batch_for_inference):
  ner_model.eval().cuda()
  device = 'cuda'
  preds = []
  for batch in tqdm(batch_for_inference, total=len(batch_for_inference)):
    batch = tuple(t.to(device) for t in batch)
    output = ner_model.inference_step(batch)
    for pred in output['best_path']:
      preds.append([ner_model.hparams.tag2name[token] for token in pred])
  return preds

def reconstruct_sentences(sentence, preds):
  reconstructed_sent = []
  tmp = []
  reconstructed_labels = []
  assert len(sentence) == len(preds)
  for token, pred in zip(sentence[1:-1], preds[1:-1]):
    if not tmp:
      tmp.append(token)
    if tmp and not token.startswith('##'):
      reconstructed_sent.extend(tmp)
      reconstructed_labels.append(pred)
      tmp = [token]
    elif tmp and token.startswith('##'):
      tmp[-1] = tmp[-1] + token[2:]
  reconstructed_sent.extend(tmp)
  return reconstructed_sent[1:], reconstructed_labels

def get_mentions(output: tuple) -> list:
  mentions = []
  tmp = []
  for token, label in list(zip(output[0], output[1])):
    if not tmp and label.startswith('B-'):
      tmp.append(token.lower())
    if tmp and label == 'O':
      mentions.append(tmp)
      tmp = []
    if tmp and label.startswith('I-'):
      tmp.append(token.lower())
  return mentions

def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1

def write_results(tokenized_texts, preds, offset_sentence_map, out_path:str):
  j = 2
  for i in tqdm(range(len(tokenized_texts))):
    assert len(tokenized_texts) == len(preds)
    output = reconstruct_sentences(tokenized_texts[i][:300], preds[i])
    mentions = get_mentions(output)
    offsets_tuples = list(offset_sentence_map[i]['offsets'].items())
    tokens = [tup[1].lower() for tup in offsets_tuples]
    if not mentions:
        #print('no mentions found')
        continue
    else:
        for mention in mentions:
            try:
                start, end = find_sub_list(mention, tokens)
                start = offsets_tuples[start][0]
                end = offsets_tuples[end][0] + len(mention[-1])
                #offset = (start, end)
                mention_ = ' '.join(entity for entity in mention)
                entry = f'T{j}\tDisposition {start} {end}\t{mention_}\n'
                with open(out_path, 'a+') as file:
                    file.write(entry)
                    j += 1
            except Exception as e:
                continue

def process_sentences(evaluation_documents: list, nlp_model, ner_model, outpath: str):
    for document in tqdm(evaluation_documents):
        with open(document) as f:
            text = f.read()
        tail = os.path.split(document)[1].split('.')[0]
        file_out = f'{tail}.ann'
        out_path = os.path.join(outpath, file_out)
        offset_sentence_map, sentences = get_offset_to_sentences_dict(
            text, nlp_model)
        
        batch_for_inference, tokenized_texts = prepare_batch_for_inference(sentences)
        preds = get_predictions(ner_model, batch_for_inference)
        write_results(tokenized_texts, preds, offset_sentence_map, out_path)
        

def parse_args(args):
    epilog = ''
    description = ''
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog)
    parser.add_argument('--config', help='Path to the config', type=str),
    parser.add_argument('--eval_data', help='Path to the eval_data', type=str)
    parser.add_argument('--outpath', help='Path to the results folder', type=str)          
    return parser.parse_args(args)


if __name__ == '__main__':
    main()
