from .train import get_model
from .val import validation
from .config import weights_file_path
import torch
from torch.utils.tensorboard import SummaryWriter
from .utils import get_tokenizer
from .pre_dataset import load_data, get_dataloader_test

def test_model(config):
    device = config["device"]
    device = torch.device(device)

    # load data and tokenizer
    dataset = load_data(config)
    tokenizer_src, tokenizer_tgt = get_tokenizer(config=config,
                                                 dataset=dataset)
    
    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    pad_id_token = tokenizer_tgt.token_to_id("[PAD]")

    # get dataloader
    test_dataloader = get_dataloader_test(config=config,
                                          dataset=dataset,
                                          tokenizer_src=tokenizer_src,
                                          tokenizer_tgt=tokenizer_tgt
    )

    model = get_model(config=config,
                        device=device,
                        src_vocab_size=src_vocab_size,
                        tgt_vocab_size=tgt_vocab_size,
                        pad_id_token=pad_id_token)
    
    writer = SummaryWriter(config["experiment_name"])
    model_filenames = weights_file_path(config=config)
    model_filename = model_filenames[-1]

    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    bleu_results = {} #[bleu <1, 2, 3, 4>][beam_size]
    for i in range(0, 4):
        bleu_results[f"Test_model_Bleu_{i + 1}"] = {}

    max_beam = config["beam_test"]

    for beam_size in range(1, max_beam + 1):
        scores_corpus = validation(model=model,
                config=config,
                tokenizer_src=tokenizer_src,
                tokenizer_tgt=tokenizer_tgt,
                validation_dataloader=test_dataloader,
                epoch=0,
                beam_size=beam_size,
                have_test=True)
        
        for i in range(len(scores_corpus)):
            bleu_results[f"Test_model_Bleu_{i + 1}"][f"Beam_size={beam_size}"] = scores_corpus[i]

    print()
    print(bleu_results)
    for i in range(0, 4):
        writer.add_scalars(f"Test_model_Bleu_{i + 1}", bleu_results[f"Test_model_Bleu_{i + 1}"], 0)
        writer.close()

def test_model_with_beam_size(config, beam_size):
    device = config["device"]
    device = torch.device(device)

    # load data and tokenizer
    dataset = load_data(config)
    tokenizer_src, tokenizer_tgt = get_tokenizer(config=config,
                                                 dataset=dataset)
    
    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    pad_id_token = tokenizer_tgt.token_to_id("[PAD]")

    # get dataloader
    test_dataloader = get_dataloader_test(config=config,
                                          dataset=dataset,
                                          tokenizer_src=tokenizer_src,
                                          tokenizer_tgt=tokenizer_tgt
    )

    model = get_model(config=config,
                        device=device,
                        src_vocab_size=src_vocab_size,
                        tgt_vocab_size=tgt_vocab_size,
                        pad_id_token=pad_id_token)
    
    writer = SummaryWriter(config["experiment_name"])
    model_filenames = weights_file_path(config=config)
    model_filename = model_filenames[-1]

    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    bleu_results = {} #[bleu <1, 2, 3, 4>][beam_size]
    for i in range(0, 4):
        bleu_results[f"Test_model_Bleu_{i + 1}"] = {}

    scores_corpus = validation(model=model,
            config=config,
            tokenizer_src=tokenizer_src,
            tokenizer_tgt=tokenizer_tgt,
            validation_dataloader=test_dataloader,
            epoch=0,
            beam_size=beam_size,
            have_test=True)
    
    for i in range(len(scores_corpus)):
        bleu_results[f"Test_model_Bleu_{i + 1}"][f"Beam_size={beam_size}"] = scores_corpus[i]

    print()
    print(bleu_results)
    for i in range(0, 4):
        writer.add_scalars(f"Test_model_Bleu_{i + 1}", bleu_results[f"Test_model_Bleu_{i + 1}"], 0)
        writer.close()