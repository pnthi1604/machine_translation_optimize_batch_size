import torch
from .beam_search import beam_search
from .train import get_model
from tokenizers import Tokenizer
from .config import weights_file_path
from .utils import create_src_mask
from .pre_dataset import clean_data, handle_lang_vi

def handle_sentence(sentence, config):
    return handle_lang_vi(sent=clean_data(text=sentence, lang=config["lang_src"]), lang=config["lang_src"], config=config)

def translate_with_beam_size(config, beam_size, sentence):
    device = config["device"]
    
    tokenizer_src = Tokenizer.from_file(config["tokenizer_file"].format(config["lang_src"]))
    tokenizer_tgt = Tokenizer.from_file(config["tokenizer_file"].format(config["lang_tgt"]))
    # print(f"{tokenizer_src = }")
    # print(f"{tokenizer_tgt = }")

    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    pad_id_token = tokenizer_tgt.token_to_id("[PAD]")

    model_filenames = weights_file_path(config=config)
    model_filename = model_filenames[-1]
    # print(f"{model_filename = }")

    model = get_model(config=config,
                      device=device,
                      src_vocab_size=src_vocab_size,
                      tgt_vocab_size=tgt_vocab_size,
                      pad_id_token=pad_id_token)

    state = torch.load(model_filename)
    model.load_state_dict(state["model_state_dict"])
    src_text = handle_sentence(sentence=sentence, config=config)
    sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)

    # print(f"{sos_token = }")
    # print(f"{eos_token = }")
    # print(f"{src_text = }")

    enc_input_tokens = tokenizer_src.encode(src_text).ids

    src = torch.cat(
        [
            sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            eos_token,
        ],
        dim=0,
    ).to(device)

    src = src.unsqueeze(0)

    src_mask = create_src_mask(src=src,
                            pad_id_token=pad_id_token,
                            device=device)
    
    with torch.no_grad():
        model.eval()
        model_out = beam_search(model=model,
                        config=config,
                        beam_size=beam_size,
                        tokenizer_src=tokenizer_src,
                        tokenizer_tgt=tokenizer_tgt,
                        src=src,
                        src_mask=src_mask)
        
        pred_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

        print(pred_text)
