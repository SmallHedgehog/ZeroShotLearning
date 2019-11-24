import numpy as np

__all__ = ['MCA']


class MCA(object):
    def __init__(self, num_classes=50):
        super(MCA, self).__init__()
        self.num_classes = num_classes

        self.classes = { c_idx: {'corrects': 0, 'nums': 0} for c_idx in range(self.num_classes)}

    def update(self, targets, preds):
        for idx in range(targets.shape[0]):
            self.classes[int(targets[idx])]['nums'] += 1

        eq_idx = np.where(targets == preds)[0]
        for idx in range(eq_idx.shape[0]):
            self.classes[int(targets[eq_idx[idx]])]['corrects'] += 1

    def calc(self):
        mca = 0.

        for c_idx in range(self.num_classes):
            mca += float(self.classes[c_idx]['corrects']) / self.classes[c_idx]['nums']

        return mca / self.num_classes
