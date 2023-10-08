import csv



class SVS_Dataset:
    def __init__(self,paths ):
        with open(paths, 'rb') as csvfile:
            reader = list(csv.DictReader(csvfile))
        self.didx=reader
        self.dalen=len(self.didx)



    def __getitem__(self, i):
        # return self.xs[i[0]], self.ys[i[0]]
        self.didx[i]

        return

    def __len__(self):
        return self.dalen

class SVC_Dataset:
    def __init__(self, ):
        pass


    def __getitem__(self, i):
        # return self.xs[i[0]], self.ys[i[0]]

        return f'svc{str(i)}'

    def __len__(self):
        return 150