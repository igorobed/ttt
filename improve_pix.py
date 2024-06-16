import numpy as np


class DelBadPix:
    def __init__(self, rgbn):
        self.r, self.g, self.b, self.n = rgbn[:, :, 0], rgbn[:, :, 1], rgbn[:, :, 2], rgbn[:, :, 3]

    @staticmethod
    def check_is(arr, i, j):
        try:
            arr[i, j]
        except:
            return 0, 0
        return 1, arr[i, j]

    def change_val(self, arr, mask):
        arr = arr.astype(np.uint32)
        i_lst, j_lst = np.nonzero(mask)
        for i, j in zip(i_lst, j_lst):
            c0, s0 = self.check_is(arr, i - 1, j - 1)
            c1, s1 = self.check_is(arr, i - 1, j + 1)
            c2, s2 = self.check_is(arr, i - 1, j)
            c3, s3 = self.check_is(arr, i, j - 1)
            c4, s4 = self.check_is(arr, i, j + 1)
            c5, s5 = self.check_is(arr, i + 1, j - 1)
            c6, s6 = self.check_is(arr, i + 1, j + 1)
            c7, s7 = self.check_is(arr, i + 1, j) 
            arr[i, j] = (s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7) / (c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7)
        return arr.astype(np.uint16)

    def del_bad_channels(self):
        max_r, max_g, max_b, max_n = (np.percentile(self.r.ravel(), 99),
                                      np.percentile(self.g.ravel(), 99),
                                      np.percentile(self.b.ravel(), 99),
                                      np.percentile(self.n.ravel(), 99))
        max_r = np.mean(self.r.ravel()[self.r.ravel() > max_r]) + 2 * np.std(self.r.ravel()[self.r.ravel() > max_r])
        max_g = np.mean(self.g.ravel()[self.g.ravel() > max_g]) + 2 * np.std(self.g.ravel()[self.g.ravel() > max_g])
        max_b = np.mean(self.b.ravel()[self.b.ravel() > max_b]) + 2 * np.std(self.b.ravel()[self.b.ravel() > max_b])
        max_n = np.mean(self.n.ravel()[self.n.ravel() > max_n]) + 2 * np.std(self.n.ravel()[self.n.ravel() > max_n])

        mask = np.bitwise_or(self.r > max_r, self.g > max_g)
        mask = np.bitwise_or(mask, self.b > max_b)
        mask = np.bitwise_or(mask, self.n > max_n)

        min_r, min_g, min_b, min_n = (np.percentile(self.r.ravel(), 1),
                                      np.percentile(self.g.ravel(), 1),
                                      np.percentile(self.b.ravel(), 1),
                                      np.percentile(self.n.ravel(), 1))

        min_r = np.mean(self.r.ravel()[self.r.ravel() < min_r]) - 2 * np.std(self.r.ravel()[self.r.ravel() < min_r])
        min_g = np.mean(self.g.ravel()[self.g.ravel() < min_g]) - 2 * np.std(self.g.ravel()[self.g.ravel() < min_g])
        min_b = np.mean(self.b.ravel()[self.b.ravel() < min_b]) - 2 * np.std(self.b.ravel()[self.b.ravel() < min_b])
        min_n = np.mean(self.n.ravel()[self.n.ravel() < min_n]) - 2 * np.std(self.n.ravel()[self.n.ravel() < min_n])

        mask = np.bitwise_or(mask, self.r < min_r)
        mask = np.bitwise_or(mask, self.g < min_g)
        mask = np.bitwise_or(mask, self.b < min_b)
        mask = np.bitwise_or(mask, self.n < min_n)

        r = self.change_val(self.r, mask)
        g = self.change_val(self.g, mask)
        b = self.change_val(self.b, mask)
        n = self.change_val(self.n, mask)

        return np.dstack((r, g, b, n))
