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

    model_weights = weights_file_path(config=config)
    if len(model_weights):
        model_filename = model_weights[-1]
    else:
        ValueError("Not have model in here")

    pad_id_token = tokenizer_src.token_to_id("[PAD]")

    model = get_model(config=config,
                      src_vocab_size=tokenizer_src.get_vocab_size(),
                      tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
                      pad_id_token=pad_id_token
    )

    state = torch.load(model_filename)
    model.load_state_dict(state["model_state_dict"])
    sentence = handle_sentence(sentence=sentence, config=config)

    model.eval()
    with torch.no_grad():
        src = tokenizer_src.encode(sentence).to(device)

    src_mask = create_src_mask(src=src,
                               pad_id_token=pad_id_token)
    
    model_out = beam_search(model=model,
                       config=config,
                       beam_size=beam_size,
                       tokenizer_src=tokenizer_src,
                       tokenizer_tgt=tokenizer_tgt,
                       src=src,
                       src_mask=src_mask)
    
    pred_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

    print(pred_text)