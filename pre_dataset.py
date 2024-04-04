import torch
from torch.utils.data import Dataset
from underthesea import word_tokenize
from pyvi import ViTokenizer
import os
from datasets import load_dataset, load_from_disk, load_from_disk
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from bs4 import BeautifulSoup
import contractions

class BilingualDataset(Dataset):

    def __init__(self, ds, src_lang, tgt_lang):
        super().__init__()
        self.ds = ds
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]

        return (src_text, tgt_text)

def clean_data(text, lang):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = (text.lower()).replace(" '", "'")
    if lang == "en":
        text = contractions.fix(text)
    if lang == "vi":
        text = text.replace("you", "u")
    return text

def handle_lang_vi(sent, lang, config):
    if lang == "vi" and config["underthesea"]:
        return word_tokenize(sent, format="text")
    elif lang == "vi" and config["pyvi"]:
        return ViTokenizer.tokenize(sent)
    return sent

def preprocess_function(config, example):
    output = {}
    output[config['lang_src']] = handle_lang_vi(clean_data(example['translation'][config['lang_src']], config["lang_src"]), lang=config["lang_src"], config=config)
    output[config['lang_tgt']] = handle_lang_vi(clean_data(example['translation'][config['lang_tgt']], config["lang_tgt"]), lang=config["lang_tgt"], config=config)
    return output

def load_data(config):
    data_path = f"{config['data_path']}"

    if "data" not in config:
        config["data"] = "mt_eng_vietnamese"
        config["subset"] = "iwslt2015-vi-en"
    elif "subset" not in config:
        config["subset"] = ""

    if not os.path.exists(data_path):
        dataset = load_dataset(config["data"], config["subset"])
        # dataset_split = dataset["train"].train_test_split(train_size=config["train_size"], seed=42)
        # dataset_split["validation"] = dataset_split.pop("test")
        # dataset_split["test"] = dataset["test"]
        # dataset_split["bleu_validation"] = dataset_split["validation"].select(range(config["num_bleu_validation"]))
        # dataset_split["bleu_train"] = dataset_split["train"].select(range(config["num_bleu_validation"]))
        # dataset_split.save_to_disk(data_path)
        dataset.save_to_disk(data_path)
        print("\nĐã lưu dataset thành công!\n")

    dataset = load_from_disk(data_path)
    print("\nĐã load dataset thành công!\n")

    map_data_path = config["map_data_path"]                                                    
    if not os.path.exists(map_data_path):
        dataset = dataset.map(
            lambda item: preprocess_function(config=config, example=item),
            remove_columns=dataset["train"].column_names,
        )

    return dataset

def filter_data(item, tokenizer_src, tokenizer_tgt, config):
    src_sent = item[config['lang_src']]
    tgt_sent = item[config['lang_tgt']]
    len_list_src_token = len(tokenizer_src.encode(src_sent).ids)
    len_list_tgt_token = len(tokenizer_tgt.encode(tgt_sent).ids)
    max_len_list = max(len_list_src_token, len_list_tgt_token)
    min_len_list = min(len_list_src_token, len_list_tgt_token)
    return max_len_list <= config["max_len"] - 4 and min_len_list > 4

def collate_fn(batch, tokenizer_src, tokenizer_tgt, pad_id_token):
    src_batch, tgt_batch, label_batch, src_text_batch, tgt_text_batch = [], [], [], [], []
    sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
    for src_text, tgt_text in batch:
        enc_input_tokens = tokenizer_src.encode(src_text).ids
        dec_input_tokens = tokenizer_tgt.encode(tgt_text).ids

        src = torch.cat(
            [
                sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                eos_token,
            ],
            dim=0,
        )

        # Add only <s> token
        tgt = torch.cat(
            [
                sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                eos_token,
            ],
            dim=0,
        )

        src_batch.append(src)
        tgt_batch.append(tgt)
        label_batch.append(label)
        src_text_batch.append(src_text)
        tgt_text_batch.append(tgt_text)

    src_batch = pad_sequence(src_batch, padding_value=pad_id_token, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=pad_id_token, batch_first=True)
    label_batch = pad_sequence(label_batch, padding_value=pad_id_token, batch_first=True)

    return {
        "encoder_input": src_batch,
        "decoder_input": tgt_batch,
        "label": label_batch,
        "src_text": src_text_batch,
        "tgt_text": tgt_text_batch,
    }

def get_dataloader(config, dataset, tokenizer_src, tokenizer_tgt):
    map_data_path = config["map_data_path"]                                                    
    if not os.path.exists(map_data_path):
        dataset = dataset.filter(lambda item: filter_data(item=item,
                                                          tokenizer_src=tokenizer_src,
                                                          tokenizer_tgt=tokenizer_tgt,
                                                          config=config))
        dataset_split = dataset["train"].train_test_split(train_size=config["train_size"], seed=42)
        dataset_split["validation"] = dataset_split.pop("test")
        dataset_split["test"] = dataset["test"]
        dataset_split["bleu_validation"] = dataset_split["validation"].select(range(config["num_bleu_validation"]))
        dataset_split["bleu_train"] = dataset_split["train"].select(range(config["num_bleu_validation"]))
        dataset_split.save_to_disk(map_data_path)
        # dataset.save_to_disk(map_data_path)
        print("\nĐã lưu map data thành công!\n")
    
    dataset = load_from_disk(map_data_path)
    print("\nĐã load map data thành công!\n")

    train_dataset = BilingualDataset(
        ds=dataset["train"],
        src_lang=config["lang_src"],
        tgt_lang=config["lang_tgt"],
    )

    validation_dataset = BilingualDataset(
        ds=dataset["validation"],
        src_lang=config["lang_src"],
        tgt_lang=config["lang_tgt"],
    )

    bleu_validation_dataset = BilingualDataset(
        ds=dataset["bleu_validation"],
        src_lang=config["lang_src"],
        tgt_lang=config["lang_tgt"],
    )

    bleu_train_dataset = BilingualDataset(
        ds=dataset["bleu_train"],
        src_lang=config["lang_src"],
        tgt_lang=config["lang_tgt"],
    )

    pad_id_token = tokenizer_tgt.token_to_id("[PAD]")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config['batch_size_train'],
                                  shuffle=True, 
                                  collate_fn=lambda batch: collate_fn(batch=batch,
                                                                      pad_id_token=pad_id_token,
                                                                      tokenizer_src=tokenizer_src,
                                                                      tokenizer_tgt=tokenizer_tgt))
    validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size_validation"],
                                       shuffle=False,
                                       collate_fn=lambda batch: collate_fn(batch=batch,
                                                                           pad_id_token=pad_id_token,
                                                                           tokenizer_src=tokenizer_src,
                                                                           tokenizer_tgt=tokenizer_tgt))
    bleu_validation_dataloader = DataLoader(bleu_validation_dataset, batch_size=1,
                                            shuffle=False,
                                            collate_fn=lambda batch: collate_fn(batch=batch,
                                                                                pad_id_token=pad_id_token,
                                                                                tokenizer_src=tokenizer_src,
                                                                                tokenizer_tgt=tokenizer_tgt))
    bleu_train_dataloader = DataLoader(bleu_train_dataset, batch_size=1,
                                            shuffle=False,
                                            collate_fn=lambda batch: collate_fn(batch=batch,
                                                                                pad_id_token=pad_id_token,
                                                                                tokenizer_src=tokenizer_src,
                                                                                tokenizer_tgt=tokenizer_tgt))

    return train_dataloader, validation_dataloader, bleu_validation_dataloader, bleu_train_dataloader

def get_dataloader_test(config, dataset, tokenizer_src, tokenizer_tgt):
    map_data_path = config["map_data_path"]                                                    
    if not os.path.exists(map_data_path):
        dataset = dataset.filter(lambda item: filter_data(item=item,
                                                          tokenizer_src=tokenizer_src,
                                                          tokenizer_tgt=tokenizer_tgt,
                                                          config=config))
        dataset.save_to_disk(map_data_path)
        print("\nĐã lưu map data thành công!\n")
    
    dataset = load_from_disk(map_data_path)
    print("\nĐã load map data thành công!\n")

    test_dataset = BilingualDataset(
        ds=dataset["test"],
        src_lang=config["lang_src"],
        tgt_lang=config["lang_tgt"],
    )

    pad_id_token = tokenizer_tgt.token_to_id("[PAD]")
    test_dataloader = DataLoader(test_dataset, batch_size=1,
                                            shuffle=False,
                                            collate_fn=lambda batch: collate_fn(batch=batch,
                                                                                pad_id_token=pad_id_token,
                                                                                tokenizer_src=tokenizer_src,
                                                                                tokenizer_tgt=tokenizer_tgt))

    return test_dataloader
