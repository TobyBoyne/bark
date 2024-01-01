class Dimension:
    def __init__(self, bnds, var_type):
        self.is_bin = False
        if var_type in ('int', 'conti'):
            lb, ub = bnds
            if (var_type == 'int') and \
               (lb == 0 and ub == 1):
                self.is_bin = True

        self.var_type = var_type
        self.bnds = bnds

    def __str__(self):
        return f"({self.var_type}, {self.bnds})"


class Space:
    def __init__(self,
        bnds,
        cat_idx=None,
        int_idx=None):

        self.cat_idx = [] if cat_idx is None else cat_idx
        self.int_idx = [] if int_idx is None else int_idx
        self.cont_idx = [idx for idx in range(len(bnds))
                         if idx not in self.cat_idx and idx not in self.int_idx]

        # define dimensions
        self.dims = []
        for idx,b in enumerate(bnds):
            if idx in self.cont_idx:
                self.dims.append(Dimension(b, 'conti'))
            elif idx in self.int_idx:
                self.dims.append(Dimension(b, 'int'))
            elif idx in self.cat_idx:
                self.dims.append(Dimension(b, 'cat'))

    def get_bounds(self):
        return [d.bnds for d in self.dims]

    def process_vals(self, X):
        pass

    def __str__(self):
        return str([str(d) for d in self.dims])