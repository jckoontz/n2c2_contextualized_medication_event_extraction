import sys
import os
import argparse
import yaml
from tqdm import tqdm
import spacy
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer
import pandas as pd
import random

from model import NER_CRF


def main(args=None):

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    config = load_config(args.config)

    nlp = spacy.load('en_core_web_sm', disable=[
                     'tagger', 'parser', 'ner', 'lemmatizer'])
    nlp.add_pipe("sentencizer")

    documents = process_data(args.note_events, nlp)

    ner_model = NER_CRF.load_from_checkpoint(config['checkpoint'],
                                             hparams_file=config['hparams'])

    process_sentences(documents, ner_model, args.outpath)


def load_config(config_path: str) -> dict:
    '''
    Load configuration file
    '''
    with open(config_path) as f:
        config = yaml.safe_load(f)
        return config


def process_data(path: str, nlp):
    notes_df = pd.read_csv(path)
    sample = round(len(notes_df['SUBJECT_ID'].unique()) * 0.0001)
    patient_ids = random.sample(
        notes_df['SUBJECT_ID'].unique().tolist(), sample)
    texts = []
    notes_df_filtered = notes_df[notes_df['SUBJECT_ID'].isin(
        patient_ids)].TEXT.tolist()
    for note in notes_df_filtered:
        doc = nlp(note)
        tmp = []
        for sent in doc.sents:
            tokens = sent.text.split()
            if tokens:
                tmp.append(tokens)
            else:
                continue
        texts.append(tmp)
    return texts


def prepare_batch_for_inference(sentences: list):
    MAX_LEN = 300
    tokenizer = AutoTokenizer.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT", do_lowercase=False)
    tokenized_texts = bert_tokenize_for_inference(sentences, tokenizer)
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    attention_masks = [[int(i > 0) for i in ii] for ii in input_ids]
    input_ids_tensors = torch.tensor(input_ids)
    attention_masks_tensors = torch.tensor(attention_masks)
    dataset = TensorDataset(input_ids_tensors, attention_masks_tensors)
    dataloader = DataLoader(dataset, batch_size=32)
    return dataloader, tokenized_texts


def bert_tokenize_for_inference(sentences: list, tokenizer) -> tuple:
    tokenized_texts = []
    print('here')
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


def write_results(tokenized_texts, preds, out_path: str):
    j = 2
    for i in tqdm(range(len(tokenized_texts))):
        output = reconstruct_sentences(tokenized_texts[i][:300], preds[i])
        mentions = get_mentions(output)
        if not mentions:
            continue
        else:
            for mention in mentions:
                try:
                    mention_ = ' '.join(entity for entity in mention)
                    entry = f'T{j}\Medication {mention_}\n'
                    with open(out_path, 'a+') as file:
                        file.write(entry)
                        j += 1
                except Exception as e:
                    continue


def process_sentences(documents: list, ner_model, outpath: str):
    for document in tqdm(documents):
        batch_for_inference, tokenized_texts = prepare_batch_for_inference(
            document)
        preds = get_predictions(ner_model, batch_for_inference)
        write_results(tokenized_texts, preds, outpath)


def parse_args(args):
    epilog = ''
    description = ''
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog)
    parser.add_argument('--config', help='Path to the config', type=str),
    parser.add_argument(
        '--note_events', help='Path to the eval_data', type=str)
    parser.add_argument(
        '--outpath', help='Path to the results folder: folder/mimic_precictions.txt', type=str)
    return parser.parse_args(args)


if __name__ == '__main__':
    main()
