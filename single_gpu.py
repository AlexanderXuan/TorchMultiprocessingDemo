import os
import sys
import torch
import torch.multiprocessing as mp

# init a unmix model
def init_model():
    model = torch.hub.load('sigsep/open-unmix-pytorch', 'umxhq')
    model.cuda()
    model.eval()
    return model

def process_func(data, model):
    est_sources = model(data)
    return est_sources.cpu().numpy()

def multi_process():
    model = init_model()
    # use model in two processes
    data_list = [torch.rand(1, 2, 44100*10)] * 2
    params = [(data.cuda(), model) for data in data_list]
    with mp.Pool(2) as pool:
        results = pool.starmap(process_func, params)
    
    return results

if __name__ == '__main__':
    model = init_model()
    # init one model and use it in two processes
    results = multi_process()
    print(len(results), results[0].shape)
    print("Done.")