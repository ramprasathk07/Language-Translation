from model import *
from dataset import *
from config import get_weights_file_path
from tqdm import tqdm 
import torchmetrics

import torch.utils.tensorboard
from  torch.utils.tensorboard import SummaryWriter

def get_model(config,vocab_seq_len,vocab_tgt_len):
    model = build_transformer(vocab_seq_len,vocab_tgt_len,config['seq_len'],config['d_model'])
    return model

def greedy_decode(model,source,source_mask,token_src,token_tgt,max_len,device):
    sos_idx = token_src.token_to_id('[SOS]')
    eos_idx = token_tgt.token_to_id('[EOS]')

    #compute the encoder output and use it for every decoder 
    encoder_out = model.encode(source,source_mask)
    decoder_inp = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_inp.size(1) == max_len:
            break

        #build mask for the target not to see future words

        decoder_mask = causal_mask(decoder_inp.size(1)).type_as(source).to(device)
        out = model.decode(encoder_out,source_mask,decoder_inp,decoder_mask)

        prob = model.proj(out[:,-1])
        _,next_word = torch.max(prob,dim = 1)
        decoder_inp = torch.cat([decoder_inp,torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)],dim = 1 )

        if next_word  == eos_idx:
            break

    return decoder_inp.squeeze(0)

def validation(model,valid_ds,token_src,token_tgt,max_len,device,print_msg,metrics = True,num_ex = 10):
    model.eval()
    count = 0
    src_txt = []
    expected_txt = []
    predicted = []

    with torch.no_grad():
        for batch in valid_ds:
            count+=1
            encoder_inp = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)   #(B,1,1,seq_len)

            assert encoder_inp.size(0) == 1,"Batch size must be 1"

            model_out = greedy_decode(model,encoder_inp,encoder_mask,token_src,token_tgt,max_len,device)

            source = batch['src_txt'][0]
            target = batch['tgt_txt'][0]
            model_out_txt = token_tgt.decode(model_out.detach().cpu().numpy())

            src_txt.append(source)
            expected_txt.append(target)
            predicted.append(model_out_txt)

            print_msg("-"*40)
            print_msg(f"SOURCE:{source}")
            print_msg(f"TARGET:{target}")
            print_msg(f"PREDICTED:{model_out_txt}")

            if count>=num_ex:
                break
    
    #Need to add metrics
    if metrics:
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected_txt)

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected_txt)

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected_txt)

    print_msg(f"BLEU:{bleu}\tWER:{wer}\tCER:{cer}")

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using Device:{device}")

    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)

    train_dl,val_dl , token_src,token_tgt = get_data(config=config)

    model = get_model(config,token_src.get_vocab_size(),token_tgt.get_vocab_size(),)
    model = model.to(device)

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(),lr = config['lr'],eps= 1e-9)

    initial_epoch = 0
    global_step = 0
    
    preload = config['preload']
    model_filename = get_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=token_src.token_to_id('[PAD]'),label_smoothing=0.1).to(device=device)

    for epoch in range(initial_epoch,config['num_epochs']):
        torch.cuda.empty_cache()
        batch_iterator = tqdm(train_dl,desc = f"Processing epoch:{epoch}")

        for batch in batch_iterator:
            model.train()
            encoder_inp = batch['encoder_input'].to(device)
            decoder_inp = batch['decoder_input'].to(device).long() 
            # decoder_inp = decoder_inp.long()
            encoder_mask = batch['encoder_mask'].to(device)   #(B,1,1,seq_len)
            decoder_mask = batch['decoder_mask'].to(device)   #(B,1,seq_len,seq_len)

            # print(f"\nencoder_inp:{encoder_inp.shape},encoder_mask:{encoder_mask.shape}\n")
            encoder_out = model.encode(encoder_inp,encoder_mask) #B,seq_len,d_model
            print(f"\encoder_out:{encoder_out.shape},encoder_mask:{encoder_mask.shape},decoder_inp:{decoder_inp.shape},decoder_mask:{decoder_mask.shape}\n")
            decoder_out = model.decode(encoder_out,encoder_mask,decoder_inp,decoder_mask) #B,seq_len,d_model

            proj_out = model.proj(decoder_out) #B,vocab_tgt

            label = batch['label'].to(device)

            loss = loss_fn(proj_out.view(-1,token_tgt.get_vocab_size()),label.view(-1))

            batch_iterator.set_postfix({f"Loss:":f"{loss.item():6.3f}"})

            writer.add_scalar('train_loass',loss.item(),global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 15 == 0:
                validation(model,val_dl,token_src,token_tgt,config['seq_len'],device,lambda msg:batch_iterator.write(msg))

    model_filename = get_weights_file_path(config,f'{epoch:02d}')
    torch.save({
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'global_step':global_step
    },model_filename)

if __name__ =='__main__':
    with open('config.yaml','rb') as f:
        onfig = yaml.safe_load(f)

    train(config=config)