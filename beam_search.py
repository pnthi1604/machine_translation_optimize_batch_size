import torch
from .utils import causal_mask
from .models.seq2seq_transformer import Seq2seqTransformer

def sequence_length_penalty(length: int, alpha: float=0.6) -> float:
    return ((5 + length) / (5 + 1)) ** alpha

def beam_search(model: Seq2seqTransformer, config, beam_size, tokenizer_src, tokenizer_tgt, src, src_mask):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    pad_id_token = tokenizer_src.token_to_id("[PAD]")
    device = config["device"]
    max_len = config["max_len"]

    encoder_output = model.encode(src=src, src_mask=src_mask)
    decoder_initial_input = torch.empty(1, 1).fill_(sos_idx).type_as(src).to(device)

    candidates = [(decoder_initial_input, 0)]

    while True:
        if all([(cand[0][-1].item() == eos_idx or cand.size(1) == max_len) for cand, _ in candidates]):
            break

        new_candidates = []

        for candidate, score in candidates:
            if candidate[0][-1].item() == eos_idx or candidate.size(-1) == max_len:
              new_candidates.append((candidate, score))
              continue
            candidate_mask = causal_mask(candidate.size(-1), device).type_as(src_mask).to(device)
            decoder_output = model.deocde(encoder_output=encoder_output,
                                          tgt=candidate,
                                          tgt_mask=candidate_mask,
                                          src_mask=src_mask)
            out = model.out(transformer_out=decoder_output)
            prob = torch.nn.functional.log_softmax(out[:, -1], dim=1)
            prob = prob / sequence_length_penalty(candidate.size(-1), alpha=0.6)
            topk_prob, topk_idx = torch.topk(prob, beam_size, dim=1)
            for i in range(beam_size):
                token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                token_prob = topk_prob[0][i].item()
                new_candidate = torch.cat([candidate, token], dim=1)
                new_candidates.append((new_candidate, score + token_prob))

        candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
        candidates = candidates[:beam_size]

    # Return the best candidate
    return candidates[0][0].squeeze()

__all__ = [
    "beam_search"
]