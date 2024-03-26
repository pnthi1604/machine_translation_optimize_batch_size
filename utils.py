import torch
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item[lang]

def get_or_build_tokenizer(config, dataset, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_tokenizer(config, dataset):
    if "train" in dataset:
        tokenizer_src = get_or_build_tokenizer(config, dataset["train"], config["lang_src"])
        tokenizer_tgt = get_or_build_tokenizer(config, dataset["train"], config["lang_tgt"])
    else:
        tokenizer_src = get_or_build_tokenizer(config, dataset, config["lang_src"])
        tokenizer_tgt = get_or_build_tokenizer(config, dataset, config["lang_tgt"])
    return tokenizer_src, tokenizer_tgt

def bleu_score(tgt_text, pred_text, max_n, func_bleu: nltk.translate.bleu_score):
    scores = []
    for j in range(1, max_n + 1):
        weights = [1 / j] * j
        scores.append(func_bleu(tgt_text, pred_text, weights))
    return scores

def sent_scores(tgt_text, pred_text, max_n=4):
    return bleu_score(tgt_text=tgt_text,
                      pred_text=pred_text,
                      max_n=max_n,
                      func_bleu=sentence_bleu)

def corpus_scores(tgt_texts, pred_texts, max_n=4):
    return bleu_score(tgt_text=tgt_texts,
                      pred_text=pred_texts,
                      max_n=max_n,
                      func_bleu=corpus_bleu)

def create_src_mask(src, pad_id_token, device):
    src_mask = (src != pad_id_token).unsqueeze(0).unsqueeze(0).permute(2, 0, 1, 3).type(torch.int64)
    return src_mask.to(device)

def causal_mask(size, device):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int64).to(device)
    return mask == 0

def create_tgt_mask(tgt, pad_id_token, device):
    return (create_src_mask(src=tgt, pad_id_token=pad_id_token, device=device) & causal_mask(tgt.size(-1), device=device)).to(device)