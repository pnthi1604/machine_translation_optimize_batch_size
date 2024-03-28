import torch
import torch.nn as nn
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from .models.seq2seq_transformer import Seq2seqTransformer
from .config.config import get_weights_file_path, weights_file_path, save_config, create_all_dic
from .utils import get_tokenizer, create_src_mask, create_tgt_mask
from .pre_dataset import load_data, get_dataloader
from .val import validation

def get_model(config, device, src_vocab_size, tgt_vocab_size, pad_id_token):
    seq2seq_transformer = Seq2seqTransformer(src_vocab_size=src_vocab_size,
                               tgt_vocab_size=tgt_vocab_size,
                               pad_id_token=pad_id_token,
                               d_model=config["d_model"],
                               num_encoder=config["num_encoder"],
                               num_decoder=config["num_decoder"],
                               h=config["nhead"],
                               dropout=config["dropout"],
                               d_ff=config["d_ff"],
                               max_len=config["max_len"]
    )

    for p in seq2seq_transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return seq2seq_transformer.to(device)

def save_model(model, epoch, global_step, optimizer, model_filename, lr_scheduler=None):
    if lr_scheduler:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'lr_scheduler_state': lr_scheduler.state_dict()
        }, model_filename)
    else:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
    
def get_lr(global_step: int, config):
  global_step = max(global_step, 1)
  return (config["d_model"] ** -0.5) * min(global_step ** (-0.5), global_step * config["warmup_steps"] ** (-1.5))

def train_model(config):
    # get config and create dictional for save model and tokenizer
    create_all_dic(config=config)

    device = config["device"]
    device = torch.device(device)
    
    # load data and tokenizer
    dataset = load_data(config)
    tokenizer_src, tokenizer_tgt = get_tokenizer(config=config,
                                                 dataset=dataset)
    
    print(f"src_vocab_size = {tokenizer_src.get_vocab_size()}")
    print(f"tgt_vocab_size = {tokenizer_tgt.get_vocab_size()}")
    
    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    pad_id_token = tokenizer_tgt.token_to_id("[PAD]")

    # get dataloader
    train_dataloader, validation_dataloader, bleu_validation_dataloader, bleu_train_dataloader = get_dataloader(config=config,
                                                                                                                dataset=dataset,
                                                                                                                tokenizer_src=tokenizer_src,
                                                                                                                tokenizer_tgt=tokenizer_tgt,
    )

    config["len_train_dataloader"] = len(train_dataloader)
    config["len_validation_dataloader"] = len(validation_dataloader)
    config["len_bleu_validation_dataloader"] = len(bleu_validation_dataloader)

    # get model
    model = get_model(config=config,
                      device=device,
                      src_vocab_size=src_vocab_size,
                      tgt_vocab_size=tgt_vocab_size,
                      pad_id_token=pad_id_token)

    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    # get optimizer Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=config["eps"], weight_decay=config["weight_decay"], betas=config["betas"])

    # preload or starting from scratch
    initial_epoch = 0
    global_step = 0
    if config["lr_scheduler"]:
        if config["lambdalr"]:
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda= lambda global_step: get_lr(global_step=global_step, config=config))
        elif config["steplr"]:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size_steplr"], gamma=config["gamma_steplr"])
    else:
        lr_scheduler = None
    preload = config["preload"]
    model_filename = (str(weights_file_path(config)[-1]) if weights_file_path(config) else None) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        if lr_scheduler:
            lr_scheduler.load_state_dict(state['lr_scheduler_state'])
    else:
        print('No model to preload, starting from scratch')

    # get Cross_entropy
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id_token, label_smoothing=config["label_smoothing"]).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()
        if config["lr_scheduler"]:
            print(f"\nlr = {lr_scheduler.get_last_lr()[0]}")

        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        train_loss = 0
        validation_loss = 0
        for batch in batch_iterator:
            src = batch['encoder_input'].to(device) # (b, seq_len)
            tgt = batch['decoder_input'].to(device) # (B, seq_len)
            src_mask = create_src_mask(src, pad_id_token, device) # (B, 1, 1, seq_len)
            tgt_mask = create_tgt_mask(tgt, pad_id_token, device) # (B, 1, seq_len, seq_len)

            encoder_output = model.encode(src=src, src_mask=src_mask)
            decoder_output = model.deocde(encoder_output=encoder_output,
                                          tgt=tgt,
                                          tgt_mask=tgt_mask,
                                          src_mask=src_mask)
            logits = model.out(transformer_out=decoder_output)
            
            label = batch['label'].to(device)


            loss = loss_fn(logits.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if config["lr_scheduler"] and config["lambdalr"]:
              lr_scheduler.step()
              cur_lr = lr_scheduler.get_last_lr()[0]
              writer.add_scalars('Learning_rate', {"lr": cur_lr}, global_step)
              writer.flush()

            global_step += 1
        
        if config["lr_scheduler"] and config["steplr"]:
            lr_scheduler.step()
            cur_lr = lr_scheduler.get_last_lr()[0]
            writer.add_scalars('Learning_rate', {"lr": cur_lr}, epoch)
            writer.flush()

        with torch.no_grad():
            model.eval()
            batch_iterator = tqdm(validation_dataloader, desc=f"Validation Loss Epoch {epoch:02d}")
            for batch in batch_iterator:
                src = batch['encoder_input'].to(device) # (b, seq_len)
                tgt = batch['decoder_input'].to(device) # (B, seq_len)
                src_mask = create_src_mask(src, pad_id_token, device) # (B, 1, 1, seq_len)
                tgt_mask = create_tgt_mask(tgt, pad_id_token, device) # (B, 1, seq_len, seq_len)

                encoder_output = model.encode(src=src, src_mask=src_mask)
                decoder_output = model.deocde(encoder_output=encoder_output,
                                            tgt=tgt,
                                            tgt_mask=tgt_mask,
                                            src_mask=src_mask)
                logits = model.out(transformer_out=decoder_output)
                
                label = batch['label'].to(device)

                loss = loss_fn(logits.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
                validation_loss += loss.item()

            writer.add_scalars("Loss", {
                "Train": train_loss / len(train_dataloader),
                "Validation": validation_loss / len(validation_dataloader),
            }, epoch)
            writer.close()

            scores_val = validation(model=model,
                                    config=config,
                                    tokenizer_src=tokenizer_src,
                                    tokenizer_tgt=tokenizer_tgt,
                                    validation_dataloader=bleu_validation_dataloader,
                                    epoch=epoch,
                                    beam_size=2)
            scores_train = validation(model=model,
                                      config=config,
                                      tokenizer_src=tokenizer_src,
                                      tokenizer_tgt=tokenizer_tgt,
                                      validation_dataloader=bleu_train_dataloader,
                                      epoch=epoch,
                                      beam_size=2)
            
            for i in range(len(scores_val)):
                writer.add_scalars(f"Bleu_{i + 1}", {
                    "train": scores_train[i],
                    "val": scores_val[i]
                }, epoch)
                
                print(f"bleu_{i + 1}_train: {scores_train[i]} - bleu_{i + 1}_val: {scores_val[i]}")

        print(f"Mean train loss: {train_loss / len(train_dataloader)}")
        print(f"Mean validation loss: {validation_loss / len(validation_dataloader)}")

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        save_model(model=model,
                   epoch=epoch,
                   global_step=global_step,
                   optimizer=optimizer,
                   model_filename=model_filename,
                   lr_scheduler=lr_scheduler)
        save_config(config=config, epoch=epoch)