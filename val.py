from tqdm import tqdm
from .beam_search import beam_search
from .utils import sent_scores, corpus_scores, create_src_mask

def validation(model, config, tokenizer_src, tokenizer_tgt, validation_dataloader, epoch, beam_size):
    device = config["device"]

    source_texts = []
    expected = []
    predicted = []

    count = 0

    batch_iterator = tqdm(validation_dataloader, desc=f"Validation Bleu Epoch {epoch:02d}")
    for batch in batch_iterator:
        src = batch['encoder_input'].to(device) # (b, seq_len)
        src_mask = create_src_mask(src, tokenizer_src.token_to_id("[PAD]"), device) # (B, 1, 1, seq_len)
        src_text = batch['src_text'][0]
        tgt_text = batch['tgt_text'][0]

        model_out = beam_search(model=model,
                                config=config,
                                beam_size=beam_size,
                                tokenizer_src=tokenizer_src,
                                tokenizer_tgt=tokenizer_tgt,
                                src=src,
                                src_mask=src_mask)
        pred_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

        source_texts.append(tokenizer_src.encode(src_text).tokens)
        expected.append([tokenizer_tgt.encode(tgt_text).tokens])
        predicted.append(tokenizer_tgt.encode(pred_text).tokens)

        count += 1
        
        if count % 20 == 0:
            print()
            print(f"{f'SOURCE: ':>12}{src_text}")
            print(f"{f'TARGET: ':>12}{tgt_text}")
            print(f"{f'PREDICTED: ':>12}{pred_text}")
            scores = sent_scores(tgt_text=[tokenizer_tgt.encode(tgt_text).tokens],
                                    pred_text=tokenizer_tgt.encode(pred_text).tokens)
            print(f'BLEU OF SENTENCE {count}')
            for i in range(0, len(scores)):
                print(f'BLEU_{i + 1}: {scores[i]}')

    scores_corpus = corpus_scores(tgt_texts=expected,
                                    pred_texts=predicted)
    
    return scores_corpus