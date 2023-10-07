from torch.utils.data.sampler import Sampler



class ssvv_ssvv_BatchSampler(Sampler):
    def __init__(self, dataset, max_batch_frames, batch_size,svs_batch_size=None, sub_indices=None,
                 num_replicas=None, rank=None,
                 required_batch_count_multiple=1, batch_by_size=True, sort_by_similar_size=True,
                 shuffle_sample=False, shuffle_batch=False, seed=0, drop_last=False) -> None:

        if svs_batch_size is None:
            svs_batch_size=batch_size//2

        self.svs_batch_size=svs_batch_size
        self.svc_batch_size = batch_size-svs_batch_size
        assert svs_batch_size<svs_batch_size

        assert svs_batch_size > 0


        self.dataset = dataset
        self.max_batch_frames = max_batch_frames
        self.batch_size = batch_size
        self.sub_indices = sub_indices
        self.num_replicas = num_replicas
        self.rank = rank
        self.required_batch_count_multiple = required_batch_count_multiple
        self.batch_by_size = batch_by_size
        self.sort_by_similar_size = sort_by_similar_size
        self.shuffle_sample = shuffle_sample
        self.shuffle_batch = shuffle_batch
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.batches = None
        self.formed = None