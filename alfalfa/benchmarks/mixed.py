import numpy as np

from ..optimizer.optimizer_utils import get_opt_core
from .base import CatSynFunc, SynFunc, preprocess_data


class CatAckley(SynFunc):
    """
    adapted from: https://arxiv.org/pdf/1911.12473.pdf"""

    int_idx = {
        0,
    }
    is_vectorised = True

    def __call__(self, x, **kwargs):
        x = np.atleast_2d(x)
        z = x[:, 1:] + x[:, :1]
        return (
            -20 * np.exp(-0.2 * np.sqrt(0.2 * np.sum(z**2)))
            - np.exp(0.2 * np.sum(np.cos(2 * np.pi * z)))
            + 20
            + np.exp(1)
            + x[:, 0]
        )

    @property
    def bounds(self):
        return [[0, 4]] + [[-3.0, 3.0] for _ in range(5)]


class PressureVessel(SynFunc):
    # adapted from: https://www.scielo.br/j/lajss/a/ZsdRkGWRVtDdHJP8WTDFFpB/?format=pdf&lang=en
    is_nonconvex = True

    def __init__(self, seed: int):
        super().__init__(seed)
        self.int_idx = {0, 1}

        def X0(x):
            return x[0] * 0.0625

        def X1(x):
            return x[1] * 0.0625

        self.ineq_constr_funcs = [
            lambda x: -X0(x) + 0.0193 * x[2],
            lambda x: -X1(x) + 0.00954 * x[3],
            lambda x: -np.pi * x[3] * x[2] ** 2 - (4 / 3) * np.pi * x[2] ** 3 + 1296000,
            # this constr. is in the reference but is not necessary
            # lambda x: x[3] - 240
        ]

    def get_model_core(self):
        # define model core
        space = self.space
        model_core = get_opt_core(space)

        # add helper vars
        x = model_core._cont_var_dict

        lb_aux, ub_aux = 1 * 0.0625, 99 * 0.0625
        X0 = model_core.addVar(lb=lb_aux, ub=ub_aux)
        model_core.addConstr(X0 == x[0] * 0.0625)

        X1 = model_core.addVar(lb=lb_aux, ub=ub_aux)
        model_core.addConstr(X1 == x[1] * 0.0625)

        # add constraints
        model_core.addConstr(-X0 + 0.0193 * x[2] <= 0)
        model_core.addConstr(-X1 + 0.00954 * x[3] <= 0)

        # add helper for cubic var
        lb2, ub2 = self.bounds[2]
        x2_squ = model_core.addVar(lb=lb2**2, ub=ub2**2)
        model_core.addConstr(x2_squ == x[2] * x[2])

        model_core.addConstr(
            -np.pi * x[3] * x2_squ - (4 / 3) * np.pi * x[2] * x2_squ + 1296000 <= 0
        )

        # this constr. is in the reference but is not necessary given the bounds
        # model_core.addConstr(x[3] - 240 <= 0)

        # set solver parameter if function is nonconvex
        model_core.Params.LogToConsole = 0
        if self.is_nonconvex:
            model_core.Params.NonConvex = 2

        model_core.update()

        return model_core

    @property
    def bounds(self):
        return [(1, 99), (1, 99), (10.0, 200.0), (10.0, 200.0)]

    @preprocess_data
    def __call__(self, x, **kwargs):
        # true vars X0 and X1 are integer multiples of 0.0625
        def X0(x):
            return x[0] * 0.0625

        def X1(x):
            return x[1] * 0.0625

        f = (
            0.6224 * x[0] * x[2] * x[3]
            + 1.7781 * X1(x) * x[2] ** 2
            + 3.1661 * x[3] * X0(x) ** 2
            + 19.84 * x[2] * X0(x) ** 2
        )
        return f

    @property
    def optimum(self):
        return 6059.715


class VAESmall(CatSynFunc):
    # adapted from: https://arxiv.org/pdf/1907.01329.pdf
    # and: https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/

    def __init__(self, seed: int):
        super().__init__(seed)
        # keep track of y for pe
        self.y = []

        self.int_idx = {idx for idx in range(1, 21)}
        self.cat_idx = {idx for idx in range(21, 32)}
        # self.cat_idx = {idx for idx in range(21, 25)}#32)}

        self.is_hier = True  # change later

        self._num_enc = self._num_dec = self._num_fc = 2

        self.ineq_constr_funcs = []

        # var keys to feat index
        self._var_keys = [
            ("learning_rate", (-4.0, -2.0)),  # 0
            # encoder layers
            ("num_enc", (0, 2)),
            ("enc_l1_out_channel_size", (2, 5)),
            ("enc_l1_stride", (1, 2)),
            ("enc_l1_padding", (0, 3)),  # 4
            ("enc_l2_out_channel_size", (3, 6)),  # 5
            ("enc_l2_stride", (1, 2)),
            ("enc_l2_padding", (0, 3)),  # 7
            # fully-connected layer
            ("num_fc_enc", (0, 2)),  # 8
            ("fc1_enc_size", (0, 15)),
            ("latent_space_size", (16, 64)),
            ("num_fc_dec", (0, 2)),  # 11
            # decoder layers
            ("dec_input", (3, 6)),  # 12
            ("num_dec", (0, 2)),
            ("dec_l1_stride", (1, 2)),
            ("dec_l1_padding", (0, 3)),
            ("dec_l1_out_padding", (0, 1)),  # 16
            ("dec_l2_in_channel_size", (2, 5)),  # 17
            ("dec_l2_stride", (1, 2)),
            ("dec_l2_padding", (0, 3)),
            ("dec_l2_out_padding", (0, 1)),  # 20
            # categorical vars
            ("enc_l1_kernel_size", (3, 5)),  # 21
            ("enc_l2_kernel_size", (3, 5)),
            ("dec_l1_kernel_size", (3, 5)),
            ("dec_l2_kernel_size", (3, 5)),  # 24
            ("enc_l1_act", ("relu", "prelu", "leaky_relu")),  # 25
            ("enc_l2_act", ("relu", "prelu", "leaky_relu")),  # 26
            ("fc_enc_l1_act", ("relu", "prelu", "leaky_relu")),  # 27
            ("fc_enc_l2_act", ("relu", "prelu", "leaky_relu")),
            ("fc_dec_l1_act", ("relu", "prelu", "leaky_relu")),
            ("fc_dec_l2_act", ("relu", "prelu", "leaky_relu")),  # 30
            ("dec_l1_act", ("relu", "prelu", "leaky_relu")),  # 31
        ]
        self._default_vals = {}

        self.bnds = [bnd for key, bnd in self._var_keys]

        from .vae_nas_utils import get_test_loss  # noqa

        self._func = get_test_loss

    def is_feas(self, x):
        return True

    def get_feas_penalty(self, x):
        return 0.0

    def has_constr(self):
        return True

    def _get_var_map(self):
        # define var_map
        var_map = {}
        for idx, var_tuple in enumerate(self._var_keys):
            key, bnd = var_tuple
            var_map[key] = idx
        return var_map

    def _get_base_constr_model(self):
        # define model core
        space = self.space
        model_core = get_opt_core(space)

        # add helper vars
        x_con = model_core._cont_var_dict
        x_cat = model_core._cat_var_dict
        var_map = self._get_var_map()

        def add_full_convo_layer(model, input, layer_idx, var_map):
            # get conv params
            # kernel is a categorical choice
            k_idx = var_map[f"enc_l{layer_idx}_kernel_size"]
            k = model.addVar(name=f"enc_l{layer_idx}_kernel_size", vtype="I")
            model.addConstr(k == 3 * x_cat[k_idx][0] + 5 * x_cat[k_idx][1])

            s = x_con[var_map[f"enc_l{layer_idx}_stride"]]
            p = x_con[var_map[f"enc_l{layer_idx}_padding"]]

            # add constr for conv output size
            conv_out = model.addVar(name=f"conv_out_{layer_idx}", vtype="I")

            model.addConstr(s * conv_out == input - k + 2 * p + s)
            return conv_out

        def add_full_deconvo_layer(model, input, layer_idx, var_map):
            # get conv params
            # kernel is a categorical choice
            k_idx = var_map[f"dec_l{layer_idx}_kernel_size"]
            k = model.addVar(name=f"dec_l{layer_idx}_kernel_size", vtype="I")
            model.addConstr(k == 3 * x_cat[k_idx][0] + 5 * x_cat[k_idx][1])

            s = x_con[var_map[f"dec_l{layer_idx}_stride"]]
            p = x_con[var_map[f"dec_l{layer_idx}_padding"]]
            o = x_con[var_map[f"dec_l{layer_idx}_out_padding"]]

            # output_padding needs to be smaller than stride or dilation
            # see pytorch docs
            model.addConstr(o + 1 <= s)

            # add constr for conv output size
            deconv_out = model.addVar(name=f"deconv_out_{layer_idx}", vtype="I")

            model.addConstr(deconv_out == (input - 1) * s + k - 2 * p + o)
            return deconv_out

        ### define encoder layers
        curr_input = 28

        # add bin vars to indicate enc layers are active
        enc_act = [
            model_core.addVar(name=f"enc_layer_act_{layer_idx}", vtype="B")
            for layer_idx in range(1, self._num_enc + 1)
        ]

        model_core.addConstr(sum(enc_act) == x_con[var_map["num_enc"]])

        for layer_idx in range(1, self._num_enc):
            model_core.addConstr(enc_act[layer_idx - 1] >= enc_act[layer_idx])

        # define layer output conditions
        for layer_idx in range(1, self._num_enc + 1):
            conv_out = add_full_convo_layer(model_core, curr_input, layer_idx, var_map)

            # compute layer out depending on whether it's active or not
            layer_out = model_core.addVar(name=f"enc_layer_out_{layer_idx}", vtype="I")

            layer_act = enc_act[layer_idx - 1]

            model_core.addConstr(
                layer_out == layer_act * conv_out + (1 - layer_act) * curr_input
            )

            curr_input = layer_out

        model_core.addConstr(curr_input >= 1)

        ### define fc layers
        fc_enc_act = [
            model_core.addVar(name=f"fc_layer_act_{layer_idx}", vtype="B")
            for layer_idx in range(1, self._num_fc + 1)
        ]

        model_core.addConstr(sum(fc_enc_act) == x_con[var_map["num_fc_enc"]])

        for layer_idx in range(1, self._num_fc):
            model_core.addConstr(fc_enc_act[layer_idx - 1] >= fc_enc_act[layer_idx])

        fc_dec_act = [
            model_core.addVar(name=f"dec_layer_act_{layer_idx}", vtype="B")
            for layer_idx in range(1, self._num_fc + 1)
        ]

        model_core.addConstr(sum(fc_dec_act) == x_con[var_map["num_fc_dec"]])

        for layer_idx in range(1, self._num_fc):
            model_core.addConstr(fc_dec_act[layer_idx - 1] >= fc_dec_act[layer_idx])

        ### define decoder layers

        # add bin vars to indicate enc layers are active
        dec_act = [
            model_core.addVar(name=f"dec_layer_act_{layer_idx}", vtype="B")
            for layer_idx in range(1, self._num_dec + 1)
        ]

        model_core.addConstr(sum(dec_act) == x_con[var_map["num_dec"]])

        for layer_idx in range(1, self._num_dec):
            model_core.addConstr(dec_act[layer_idx - 1] >= dec_act[layer_idx])

        curr_input = 7
        for layer_idx in range(1, self._num_dec + 1):
            deconv_out = add_full_deconvo_layer(
                model_core, curr_input, layer_idx, var_map
            )

            # compute layer out depending on whether it's active or not
            layer_out = model_core.addVar(name=f"dec_layer_out_{layer_idx}", vtype="I")

            layer_act = dec_act[layer_idx - 1]

            model_core.addConstr(
                layer_out == layer_act * deconv_out + (1 - layer_act) * curr_input
            )

            curr_input = layer_out

        # two outcomes according to paper:
        # 1. output should be 28 if deconvolutions are active
        # 2. output should be params['dec_input'] * 7 * 7 = 28 * 28 if no deconv is active
        model_core.addConstr((dec_act[0] == 1) >> (curr_input == 28))
        model_core.addConstr((dec_act[0] == 0) >> (x_con[var_map["dec_input"]] == 4))

        # REMOVE later
        # model_core.addConstr(x_con[var_map['num_dec']] == 0)

        self._enc_act = enc_act
        self._fc_enc_act = fc_enc_act
        self._fc_dec_act = fc_dec_act
        self._dec_act = dec_act

        model_core.update()
        return model_core

    def get_model_core(self, size_is_cat=False):
        model_core = self._get_base_constr_model()
        model_core.Params.LogToConsole = 0
        model_core.Params.NonConvex = 2

        # add hierarchical constr
        x_con = model_core._cont_var_dict
        x_cat = model_core._cat_var_dict
        var_map = self._get_var_map()

        # set default values for inactive encoder layers
        for layer_idx in range(1, self._num_enc + 1):
            lb = self._var_keys[var_map[f"enc_l{layer_idx}_out_channel_size"]][1][0]
            model_core.addConstr(
                (self._enc_act[layer_idx - 1] == 0)
                >> (x_con[var_map[f"enc_l{layer_idx}_out_channel_size"]] <= lb)
            )

            lb = self._var_keys[var_map[f"enc_l{layer_idx}_stride"]][1][0]
            model_core.addConstr(
                (self._enc_act[layer_idx - 1] == 0)
                >> (x_con[var_map[f"enc_l{layer_idx}_stride"]] <= lb)
            )

            lb = self._var_keys[var_map[f"enc_l{layer_idx}_padding"]][1][0]
            model_core.addConstr(
                (self._enc_act[layer_idx - 1] == 0)
                >> (x_con[var_map[f"enc_l{layer_idx}_padding"]] <= lb)
            )

            ## add categorical constraints
            model_core.addConstr(
                (self._enc_act[layer_idx - 1] == 0)
                >> (x_cat[var_map[f"enc_l{layer_idx}_kernel_size"]][0] == 1)
            )

            model_core.addConstr(
                (self._enc_act[layer_idx - 1] == 0)
                >> (x_cat[var_map[f"enc_l{layer_idx}_act"]][0] == 1)
            )

        # set default values for inactive fc layers
        for layer_idx in range(1, self._num_fc + 1):
            if layer_idx == 1:
                # number of nodes are only set for first enc / dec layers
                lb = self._var_keys[var_map["fc1_enc_size"]][1][0]
                model_core.addConstr(
                    (self._fc_enc_act[layer_idx - 1] == 0)
                    >> (x_con[var_map["fc1_enc_size"]] <= lb)
                )

            ## add categorical constraints
            model_core.addConstr(
                (self._fc_enc_act[layer_idx - 1] == 0)
                >> (x_cat[var_map[f"fc_enc_l{layer_idx}_act"]][0] == 1)
            )

            model_core.addConstr(
                (self._fc_dec_act[layer_idx - 1] == 0)
                >> (x_cat[var_map[f"fc_dec_l{layer_idx}_act"]][0] == 1)
            )

        # set default values for inactive decoder layers
        for layer_idx in range(1, self._num_dec + 1):
            if layer_idx == 2:
                # input channels only relevant if layer 2
                lb = self._var_keys[var_map[f"dec_l{layer_idx}_in_channel_size"]][1][0]
                model_core.addConstr(
                    (self._dec_act[layer_idx - 1] == 0)
                    >> (x_con[var_map[f"dec_l{layer_idx}_in_channel_size"]] <= lb)
                )

            lb = self._var_keys[var_map[f"dec_l{layer_idx}_stride"]][1][0]
            model_core.addConstr(
                (self._dec_act[layer_idx - 1] == 0)
                >> (x_con[var_map[f"dec_l{layer_idx}_stride"]] <= lb)
            )

            lb = self._var_keys[var_map[f"dec_l{layer_idx}_padding"]][1][0]
            model_core.addConstr(
                (self._dec_act[layer_idx - 1] == 0)
                >> (x_con[var_map[f"dec_l{layer_idx}_padding"]] <= lb)
            )

            lb = self._var_keys[var_map[f"dec_l{layer_idx}_out_padding"]][1][0]
            model_core.addConstr(
                (self._dec_act[layer_idx - 1] == 0)
                >> (x_con[var_map[f"dec_l{layer_idx}_out_padding"]] <= lb)
            )

            ## add categorical constraints
            model_core.addConstr(
                (self._dec_act[layer_idx - 1] == 0)
                >> (x_cat[var_map[f"dec_l{layer_idx}_kernel_size"]][0] == 1)
            )

            if layer_idx == 1:
                model_core.addConstr(
                    (self._dec_act[layer_idx - 1] == 0)
                    >> (x_cat[var_map[f"dec_l{layer_idx}_act"]][0] == 1)
                )

        model_core.Params.NonConvex = 2
        model_core.update()

        return model_core

    @preprocess_data
    def __call__(self, x, **kwargs):
        temp_dict = dict()

        for idx, key_tuple in enumerate(self._var_keys):
            key, bnd = key_tuple

            if key == "learning_rate":
                temp_dict[key] = 10.0 ** x[idx]
            elif key == "fc1_enc_size":
                temp_dict[key] = 64 * x[idx]
            elif key.split("_")[-2] == "channel" or key == "dec_input":
                temp_dict[key] = 2 ** x[idx]
            else:
                temp_dict[key] = x[idx]

        temp_dict.update(self._default_vals)

        try:
            f = self._func(temp_dict)
        except ValueError:
            f = max(self.y) if self.y else 500

        self.y.append(f)
        return f
