


class TSVS_Dataset:
    def __init__(self, ):
        pass


    def __getitem__(self, i):
        # return self.xs[i[0]], self.ys[i[0]]

        return f'svs{str(i)}'

    def __len__(self):
        return 100

class TSVC_Dataset:
    def __init__(self, ):
        pass


    def __getitem__(self, i):
        # return self.xs[i[0]], self.ys[i[0]]

        return f'svc{str(i)}'

    def __len__(self):
        return 150