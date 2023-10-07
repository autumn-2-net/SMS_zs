from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader




class SVS_Dataset:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __getitem__(self, i):
        # return self.xs[i[0]], self.ys[i[0]]

        return self.xs[i], self.ys[i]

    def __len__(self):
        return len(self.xs)

class SVC_Dataset:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __getitem__(self, i):
        # return self.xs[i[0]], self.ys[i[0]]

        return self.xs[i], self.ys[i]

    def __len__(self):
        return len(self.xs)

class MIX_Dataset(Dataset):
    def __init__(self, svs_data_set, svc_data_set):
        self.svs_data_set = svs_data_set
        self.svc_data_set = svc_data_set
        self.num_for_svs=len(self.svs_data_set)
        self.num_for_svc = len(self.svc_data_set)
        self.mix_num= self.num_for_svs+self.num_for_svc

    def __getitem__(self, i):


        if i <self.num_for_svs:
            return self.svs_data_set[i]
        else:
            return self.svc_data_set[i-self.num_for_svs]


    def __len__(self):
        return self.mix_num

    def get_child_data_set_num(self):
        return {'svs':self.num_for_svs,'svc':self.num_for_svc}

class ssvv_ssvv_BatchSampler(Sampler):
    def __init__(self, dataset, batch_size,svs_batch_size=None,
                 num_replicas=None, rank=None,

                 shuffle=False,seed=0, drop_last=False) -> None:

        if svs_batch_size is None:
            svs_batch_size=batch_size//2

        self.svs_batch_size=svs_batch_size
        self.svc_batch_size = batch_size-svs_batch_size
        assert svs_batch_size<svs_batch_size

        assert svs_batch_size > 0


        self.dataset = dataset

        self.batch_size = batch_size

        self.num_replicas = num_replicas
        self.rank = rank

        self.shuffle = shuffle

        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.batches = None
        self.formed = None