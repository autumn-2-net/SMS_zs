import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader




class TSVS_Dataset:
    def __init__(self, ):
        pass


    def __getitem__(self, i):
        # return self.xs[i[0]], self.ys[i[0]]

        return f'svs{str(i)}'

    def __len__(self):
        return 10

class TSVC_Dataset:
    def __init__(self, ):
        pass


    def __getitem__(self, i):
        # return self.xs[i[0]], self.ys[i[0]]

        return f'svc{str(i)}'

    def __len__(self):
        return 50

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

class ssvvsc_BatchSampler(Sampler):
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
        nums=self.dataset.get_child_data_set_num()
        self.num_for_svs=nums['svs']
        self.num_for_svc = nums['svc']

        self.batch_size = batch_size

        self.num_replicas = num_replicas
        self.rank = rank

        self.shuffle = shuffle

        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def update_epoch(self):
        self.epoch += 1



    def build_batch(self):
        svs_npad = self.num_for_svs %self.svs_batch_size
        if svs_npad != 0:
            svs_pad = self.svs_batch_size - svs_npad

        svc_npad = self.num_for_svc %self.svc_batch_size
        if svc_npad != 0:
            svc_pad = self.svc_batch_size - svc_npad

        if self.drop_last:
            svs_bnum=self.num_for_svs //self.svs_batch_size
            svc_bnum = self.num_for_svc //self.svc_batch_size
        else:
            svs_bnum=self.num_for_svs //self.svs_batch_size
            svc_bnum = self.num_for_svc //self.svc_batch_size
            if svs_npad != 0:
                svs_bnum=svs_bnum+1
            if svc_npad != 0:
                svc_bnum=svc_bnum+1


        batchlist=[]   # svc 数据前置

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        # indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if svs_bnum>svc_bnum:
            svsidl=torch.randperm(self.num_for_svs, generator=g).tolist()
            tpmlist=[]
            tpmlist2 = []
            if self.drop_last and svs_npad != 0:
                svsidl=svsidl[:-svs_npad]
            else:
                if svs_npad != 0:
                    svsidl = svsidl+torch.randperm(self.num_for_svs, generator=g).tolist()[:svs_pad]
            sxcl=[]
            for i in svsidl:
                if len(sxcl) ==self.svs_batch_size:
                    tpmlist.append(sxcl.copy())
                    sxcl=[]
                sxcl.append(i)
            tpmlist.append(sxcl.copy())

            svsblen=len(tpmlist)

            svcnumitem=svsblen*self.svs_batch_size

            itspad=svcnumitem%self.num_for_svc
            itsnpad=self.num_for_svc-itspad
            if itspad!=0:
                svccpac=svcnumitem//self.num_for_svc +1
            else:
                svccpac = svcnumitem // self.num_for_svc

            svcidl=[]

            for i in range(svccpac):
                svcidl=svcidl+(torch.randperm(self.num_for_svc, generator=g)+self.num_for_svs).tolist()

            if itspad!=0:
                svcidl=svcidl[:-itsnpad]
            svc_ctpm=[]
            for i in svcidl:
                if len(svc_ctpm) == self.svc_batch_size:
                    tpmlist2.append(svc_ctpm.copy())
                    svc_ctpm = []
                svc_ctpm.append(i)
            tpmlist2.append(svc_ctpm.copy())

            assert len(tpmlist)==len(tpmlist2)

            for svcX,svsX in zip(tpmlist2,tpmlist2):
                svcX:list
                svsX:list
                batchlist.append(svcX+svsX)



