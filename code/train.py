import json
import math
import os
import shutil
from typing import Dict
from typing_extensions import Annotated

import optuna
import typer
import torch
import pandas as pd
import transformers as hf
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import CharDelimiterSplit, Sequence, Digits, Punctuation
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedTokenizerFast


def train_tokenizer(input_file: str, vocab_size: int, log_dir: str):
    df = pd.read_csv(input_file)
    tmp_path = os.path.join(log_dir, 'tmp.txt')
    outfile = os.path.join(log_dir, 'tokenizer.json')
    with open('data/tmp.txt', 'w') as fi:
        fi.write('text\n')
        for biln in df.biln:
            if not isinstance(biln, float):
                fi.write(f"{biln}\n")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "\n", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = Sequence([CharDelimiterSplit('\n'), CharDelimiterSplit("-"), Digits(), Punctuation()])
    tokenizer.train([tmp_path], trainer)
    tokenizer.save(outfile)
    return tokenizer


def run_hpo(log_dir: str, data_path: str, overwrite: Annotated[bool, typer.Option("--overwrite")] = False):
    global counter, prev_loss
    if os.path.isdir(log_dir) and not overwrite:
        raise RuntimeError("Directory already exists")
    elif os.path.isdir(log_dir) and overwrite:
        shutil.rmtree(log_dir)

    db_path = os.path.join(data_path, 'biln_db.csv')
    tokenizer_path = os.path.join(log_dir, 'tokenizer.json')

    study = optuna.create_study(direction='minimize')
    counter = 0
    prev_loss = math.inf

    def optim_objective(trial: optuna.Trial):
        global counter, prev_loss
        tb_writer = SummaryWriter(os.path.join(log_dir, f"bilnLM_{counter}"))

        num_attention_heads = trial.suggest_int("num_attention_heads",
                                                low=1, high=64)
        vocab_size = trial.suggest_int('vocab_size', low=1e2, high=3000)
        model_config_hparams = {
            "position_embedding_type": trial.suggest_categorical(
                'position_embedding_type', ['RoPE', 'absolute', 'relative_key',
                                            'relative_key_query']),
            "num_hidden_layers": trial.suggest_int('num_hidden_layers', low=1,
                                                   high=32),
            "num_attention_heads": num_attention_heads,
            "hidden_size": trial.suggest_int(
                "hidden_size", low=2, high=16) * num_attention_heads
        }
        tokenizer = train_tokenizer(db_path, vocab_size, log_dir)
        lr = trial.suggest_float("learning_rate", low=1e-7, high=1e-1,
                                 log=True)
        model, history = train_model(model_config_hparams, counter, log_dir,
                                     lr=lr, tokenizer, tb_writer=tb_writer)
        loss = history['eval_loss'].min()
        model_config_hparams['vocab_size'] = vocab_size
        model_config_hparams['learning_rate'] = lr
        model_config_hparams['model_size'] = sum(p.numel()
                                                 for p in model.parameters())
        tb_writer.add_hparams(model_config_hparams, {"hparam/loss": loss,
                                                     "hparam/run": counter})
        counter += 1
        if loss < prev_loss:
            prev_loss = loss
            tokenizer.save(os.path.join(log_dir, 'best_tokenizer.json'))
            torch.save(model.state_dict(),
                       os.path.join(log_dir, 'best_model_st_dict.pt'))
            json.dump(model_config_hparams,
                      open(os.path.join(log_dir, 'best_model_hparams.json'),
                           'w'), indent=2)
        return loss

    study.optimize(optim_objective, n_trials=30)


def train_model(model_config: dict, counter: int, log_dir: str, lr: float,
                tokenizer, tb_writer=SummaryWriter):
    # tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/tokenizer.json", padding=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]", 'mask_token': "[MASK]"})
    # tokenizer.save_pretrained(os.path.join(log_dir, 't'))
    model_config = hf.EsmConfig(vocab_size=tokenizer.vocab_size,
                                vocab_list=tokenizer.vocab,
                                pad_token_id=3, **model_config)
    model = hf.AutoModelForMaskedLM.from_config(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params / 1e6 < 1e3:
        print(f'Number of model parameters are: {n_params/1e6:.1f} M')
    else:
        print(f'Number of model parameters are: {n_params/1e9:.1f} B')

    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [line for line in examples["text"]
                            if len(line) > 0 and not line.isspace()]
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
        )
    ds = load_dataset(path='data', data_files=['tmp.txt'])
    ds = ds['train'].train_test_split(test_size=0.1)
    ds = ds.map(
        tokenize_function,
        batched=True,
        num_proc=2,
        # remove_columns=["text"],
    )
    log_strategy = "epoch"
    hf_args = hf.TrainingArguments(
            output_dir=os.path.join(log_dir, f"bilnLM_{counter}"),
            learning_rate=lr,
            num_train_epochs=10,
            greater_is_better=False,
            eval_accumulation_steps=1,
            load_best_model_at_end=True,
            evaluation_strategy=log_strategy,
            auto_find_batch_size=True,
            save_strategy=log_strategy,
            save_total_limit=1,
            report_to='tensorboard'
        )
    trainer = hf.Trainer(
        args=hf_args,
        model=model,
        data_collator=hf.DataCollatorForLanguageModeling(tokenizer),
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        tokenizer=tokenizer,
        callbacks=[hf.integrations.TensorBoardCallback(tb_writer),
                   hf.EarlyStoppingCallback(early_stopping_patience=3)]
    )
    trainer.train()
    trainer._load_best_model()
    return trainer.model, pd.DataFrame(trainer.state.log_history)


if __name__ == '__main__':
    typer.run(run_hpo)
