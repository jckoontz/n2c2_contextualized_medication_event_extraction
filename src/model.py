from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch
from torchcrf import CRF
import pytorch_lightning as pl
from seqeval.metrics import accuracy_score, f1_score

from transformers import BertModel


class NER_CRF(pl.LightningModule):
    def __init__(self, model, dropout, num_classes, total_steps, tags_vals, tag2name):
        super(NER_CRF, self).__init__()
        self.bert = BertModel.from_pretrained(model)
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.crf = CRF(num_classes, batch_first=True)
        self.total_steps = total_steps
        self.tags_vals = tags_vals
        self.tag2name = tag2name
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        input_ids, attention_masks, labels = batch
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_masks)
        output = self.dropout(output[0])
        logits = self.out(output)
        loss = -self.crf(logits, labels,
                         mask=attention_masks.bool()) / float(len(batch))
        self.log("performance", {"train_loss": loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_masks, labels = batch
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_masks)
        output = self.dropout(output[0])
        logits = self.out(output)
        loss = -self.crf(logits, labels,
                         mask=attention_masks.bool()) / float(len(batch))
        best_path = self.crf.decode(logits, mask=attention_masks.bool())
        pred, true = [], []
        for path, mask, label in zip(best_path, attention_masks, labels):
            idx = len(torch.where(mask.bool() == True)[
                      0].detach().cpu().numpy().tolist())
            true_ = label[:idx].detach().cpu().numpy().tolist()
            pred.append([self.tag2name[token] for token in path[1:-1]])
            true.append([self.tag2name[token] for token in true_][1:-1])
        accuracy = accuracy_score(true, pred)
        f1 = f1_score(true, pred)
        self.log("performance", {"val_loss": loss, "val_acc": accuracy,
                                 'val_f1': f1}, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids, attention_masks, labels = batch
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_masks)
        output = self.dropout(output[0])
        logits = self.out(output)
        loss = -self.crf(logits, labels,
                         mask=attention_masks.bool()) / float(len(batch))
        best_path = self.crf.decode(logits, mask=attention_masks.bool())
        pred, true = [], []
        for path, mask, label in zip(best_path, attention_masks, labels):
            idx = len(torch.where(mask.bool() == True)[
                      0].detach().cpu().numpy().tolist())
            true_ = label[:idx].detach().cpu().numpy().tolist()
            pred.append([self.tag2name[token] for token in path[1:-1]])
            true.append([self.tag2name[token] for token in true_][1:-1])
        accuracy = accuracy_score(true, pred)
        f1 = f1_score(true, pred)
        self.log("performance", {"test_loss": loss, "test_acc": accuracy,
                                 'test_f1': f1}, prog_bar=True)

    def predict_tags(self, batch):
        input_ids, attention_masks, labels = batch
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_masks)
        output = self.dropout(output[0])
        logits = self.out(output)
        best_path = self.crf.decode(logits, mask=attention_masks.bool())
        pred, true = [], []

        for path, mask, label in zip(best_path, attention_masks, labels):
            idx = len(torch.where(mask.bool() == True)[
                      0].detach().cpu().numpy().tolist())
            true_ = label[:idx].detach().cpu().numpy().tolist()
            pred.append([self.tag2name[token] for token in path[1:-1]])
            true.append([self.tag2name[token] for token in true_][1:-1])

        return pred, true

    def inference_step(self, batch):
        input_ids, attention_masks = batch
        output = self.bert(input_ids=input_ids,
                           attention_mask=attention_masks)
        output = self.dropout(output[0])
        logits = self.out(output)
        best_path = self.crf.decode(logits, mask=attention_masks.bool())
        return {'best_path': best_path, 'attention_masks': attention_masks}

    def configure_optimizers(self):

        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=3e-5,
            eps=1e-8
        )

        total_steps = self.total_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        scheduler = {"scheduler": scheduler,
                     "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]
