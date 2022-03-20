import sys
import os
import argparse
import yaml
import pytorch_lightning as pl
from tqdm import tqdm
from data_module import NERDataModule
from model import NER_CRF
from seqeval.metrics import accuracy_score, f1_score, classification_report


def main(args=None):

    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    config = load_config(args.config)
    
    ner_data = NERDataModule(config)

    model_config = {
    "pretrained_model": config['pretrained_model'],
    "num_classes": len(ner_data.tags_vals),
    "dropout": config['dropout'], 
    'batch_size': config['batch_size'], 
    'num_steps': len(ner_data.train_data) * 3}

    ner_crf = NER_CRF(model_config['pretrained_model'], model_config['dropout'], model_config['num_classes'], 
                  model_config['num_steps'], ner_data.tags_vals, ner_data.tag2name)
                  
    trainer = pl.Trainer(default_root_dir=config['trained_models'], gpus=config['gpus'], max_epochs=config['max_epochs'], progress_bar_refresh_rate=20)
    
    trainer.fit(ner_crf, ner_data.train_dataloader(), ner_data.val_dataloader())

    evaluate(ner_crf, ner_data)


def load_config(config_path: str) -> dict:
    '''
    Load configuration file
    '''
    with open(config_path) as f:
            config = yaml.safe_load(f)
            return config


def evaluate(ner_crf, ner_data):
    true, pred = [], []
    ner_crf.eval().cuda()
    device = 'cuda'
    for batch in tqdm(ner_data.val_dataloader(), total=len(ner_data.val_dataloader())):
        batch = tuple(t.to(device) for t in batch)
        output = ner_crf.predict_tags(batch)
        pred.extend(output[0])
        true.extend(output[1])
    print(classification_report(true, pred))
    print(accuracy_score(true, pred))
    print(f1_score(true, pred))


def parse_args(args):
    epilog = ''
    description = ''
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog)
    parser.add_argument('--config', help='Path to the config', type=str)
    return parser.parse_args(args)


if __name__ == '__main__':
    main()