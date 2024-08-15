import torch 
from model import *
from dataset import *

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

def beam_search(model,beam_size,source,source_mask,token_src,token_tgt,max_len,device):
    sos_idx = token_src.token_to_id('[SOS]')
    eos_idx = token_tgt.token_to_id('[EOS]')

    encoder_out = model.encode(source,source_mask)
    decoder_inp = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)

    candidates = [(decoder_inp, 1)]

    while True:
        if any([c.size(1)==max_len for c,_ in candidates]):
            break
        
        new_cands = []

        for candidate,score in new_cands:
            if candidate[0][-1].item() == eos_idx:
                continue 
            
            candidate_mask = causal_mask(candidate.size(1)).type_as(source_mask).to(device)

            out = model.decode(encoder_out,source_mask,candidate,candidate_mask)

            prob = model.proj(out[:,-1])
            top_prob,top_idx = torch.topk(prob,beam_size,dim=1)

            for i in range(beam_size):
                token = top_idx[0][i].unsqueeze(0).unsqueeze(0)
                token_prob = top_prob[0][i].item()
                new_cands = torch.cat([candidate, token], dim=1)
                new_cands.append((new_cands, score + token_prob))

        candidates = sorted(new_cands, key=lambda x: x[1], reverse=True)
        candidates = candidates[:beam_size]

        if all([cand[0][-1].item() == eos_idx for cand, _ in candidates]):
            break

    return candidates[0][0].squeeze()



        



