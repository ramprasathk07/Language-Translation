from model import *
from dataset import *
from config import get_weights_file_path
from tqdm import tqdm 

import torch.utils.tensorboard
from  torch.utils.tensorboard import SummaryWriter

def get_model(vocab_seq_len,vocab_tgt_len):
    model = build_transformer(vocab_seq_len,vocab_tgt_len,config['seq_len'],config['d_model'])

    return model

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

    if config['preload']:
        model_filename = get_weights_file_path(config,config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state('optimizer'))
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=token_src.token_to_id('[PAD]'),label_smoothing=0.1).to(device=device)

    for epoch in range(initial_epoch,config['num_epochs']):
        model.train()

        batch_iterator = tqdm(train_dl,desc = f"Processing epoch:{epoch}")

        for batch in batch_iterator:
            encoder_inp = batch['encoder_input'].to(device)
            decoder_inp = batch['decoder_input'].to(device)

            encoder_mask = batch['encoder_mask'].to(device)   #(B,1,1,seq_len)
            decoder_mask = batch['decoder_mask'].to(device)   #(B,1,seq_len,seq_len)

            

