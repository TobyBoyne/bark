# script based on: https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/

# import all libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# define a Conv VAE
class ConvVAE(nn.Module):
    def __init__(self, params):
        super(ConvVAE, self).__init__()

        self.verbose = False

        # print all parameter choices
        print("** params values **")
        for par in params:
            print(f"  {par}: {params[par]}")

        # setup available activation functions
        self.act_funcs = {
            "relu": nn.ReLU,
            "prelu": nn.PReLU,
            "leaky_relu": nn.LeakyReLU,
        }
        self.act_map = lambda key: self.act_funcs[params[key]]()

        # encoder
        self.enc = nn.ModuleList()
        curr_input = 1
        for idx in range(1, params["num_enc"] + 1):
            self.enc.append(
                nn.Conv2d(
                    in_channels=curr_input,
                    out_channels=params[f"enc_l{idx}_out_channel_size"],
                    kernel_size=params[f"enc_l{idx}_kernel_size"],
                    stride=params[f"enc_l{idx}_stride"],
                    padding=params[f"enc_l{idx}_padding"],
                )
            )
            curr_input = params[f"enc_l{idx}_out_channel_size"]

        # encoder fully connected
        self.fc_enc = nn.ModuleList()
        for idx in range(1, params["num_fc_enc"] + 1):
            if idx == 1:
                self.fc_enc.append(nn.Linear(curr_input, params[f"fc{idx}_enc_size"]))
                curr_input = params[f"fc{idx}_enc_size"]
            else:
                self.fc_enc.append(
                    nn.Linear(curr_input, 2 * params["latent_space_size"])
                )
                curr_input = 2 * params["latent_space_size"]

        # latent space layer
        self.dec_input = params["dec_input"]
        num_last_fc = self.dec_input * 7 * 7
        self.fc_dec = nn.ModuleList()

        if params["num_fc_dec"] == 0:
            # no fully-connected layers in the decoder are active
            self.fc_mu = nn.Linear(curr_input, num_last_fc)
            self.fc_log_var = nn.Linear(curr_input, num_last_fc)
        else:
            # at least one fully connected layer is active
            self.fc_mu = nn.Linear(curr_input, params["latent_space_size"])
            self.fc_log_var = nn.Linear(curr_input, params["latent_space_size"])
            curr_input = params["latent_space_size"]

            ## decoder fully connected
            if params["num_fc_dec"] == 2:
                self.fc_dec.append(
                    nn.Linear(curr_input, 2 * params["latent_space_size"])
                )
                curr_input = 2 * params["latent_space_size"]

            self.fc_dec.append(nn.Linear(curr_input, num_last_fc))

        # decoder
        curr_input = self.dec_input
        self.dec = nn.ModuleList()

        if params["num_dec"] == 2:
            self.dec.append(
                nn.ConvTranspose2d(
                    in_channels=curr_input,
                    out_channels=params["dec_l2_in_channel_size"],
                    kernel_size=params["dec_l1_kernel_size"],
                    stride=params["dec_l1_stride"],
                    padding=params["dec_l1_padding"],
                    output_padding=params["dec_l1_out_padding"],
                )
            )
            curr_input = params["dec_l2_in_channel_size"]

            self.dec.append(
                nn.ConvTranspose2d(
                    in_channels=curr_input,
                    out_channels=1,
                    kernel_size=params["dec_l2_kernel_size"],
                    stride=params["dec_l2_stride"],
                    padding=params["dec_l2_padding"],
                    output_padding=params["dec_l2_out_padding"],
                )
            )

        if params["num_dec"] == 1:
            self.dec.append(
                nn.ConvTranspose2d(
                    in_channels=curr_input,
                    out_channels=1,
                    kernel_size=params["dec_l1_kernel_size"],
                    stride=params["dec_l1_stride"],
                    padding=params["dec_l1_padding"],
                    output_padding=params["dec_l1_out_padding"],
                )
            )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def forward(self, x):
        # encoding
        for idx, layer in enumerate(self.enc):
            x = self.act_map(f"enc_l{idx + 1}_act")(layer(x))

        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)

        if self.verbose:
            print(f"  -> after enc: {x.shape}")

        # encoding fc
        for idx, layer in enumerate(self.fc_enc):
            x = self.act_map(f"fc_enc_l{idx + 1}_act")(layer(x))

        if self.verbose:
            print(f"  -> after enc fc: {x.shape}")

        # latent representation
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        z = self.reparameterize(mu, log_var)
        x = z

        if self.verbose:
            print(f"  -> after latent: {x.shape}")

        # decoding fc
        if self.dec:
            for idx, layer in enumerate(self.fc_dec):
                x = self.act_map(f"fc_dec_l{idx + 1}_act")(layer(x))
        else:
            if self.fc_dec:
                for idx, layer in enumerate(self.fc_dec[:-1]):
                    x = self.act_map(f"fc_dec_l{idx + 1}_act")(layer(x))
                x = self.fc_dec[-1](x)

        if self.verbose:
            print(f"  -> after dec fc: {x.shape}")

        # decoding
        if self.dec:
            x = x.view(batch, self.dec_input, 7, 7)
            if self.verbose:
                print(f"  -> after view: {x.shape}")

            for idx, layer in enumerate(self.dec[:-1]):
                x = self.act_map(f"dec_l{idx + 1}_act")(layer(x))
                if self.verbose:
                    print(f"  -> after dec: {x.shape}")

            x = self.dec[-1](x)
            x = x.view(batch, 1, 28, 28)
            if self.verbose:
                print(f"  -> after dec: {x.shape}")
        else:
            x = x.view(batch, 1, 28, 28)

        if self.verbose:
            print(f"  -> after dec: {x.shape}")

        # reconstruction
        reconstruction = torch.sigmoid(x)

        if self.verbose:
            print(f"  -> after final size: {x.shape}")

        # check if the output image has the correct size
        x_out, y_out = reconstruction.shape[2:]

        if (
            reconstruction.shape[0] != batch
            or reconstruction.shape[1] != 1
            or x_out != 28
            or y_out != 28
        ):
            return None
        return reconstruction, mu, log_var


def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def validate(model, dataloader, dataset, device, criterion):
    model.eval()
    running_loss = 0.0
    running_rec_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(
            enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)
        ):
            data = data[0]
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            running_rec_loss += bce_loss

            # save the last batch input and output of every epoch
            if i == int(len(dataset) / dataloader.batch_size) - 1:
                recon_images = reconstruction
    val_loss = running_loss / len(dataloader.dataset)
    rec_loss = running_rec_loss / len(dataloader.dataset)
    return val_loss, recon_images, rec_loss


def train(model, dataloader, dataset, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(
        enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)
    ):
        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / len(dataloader.dataset)
    return train_loss


def get_test_loss(params):
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    # store old seed and set new seeds
    old_np_seed = np.random.get_state()
    old_torch_seed = torch.get_rng_state()
    np.random.seed(101)
    torch.manual_seed(101)

    # matplotlib.style.use('ggplot')

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")
    # initialize the model
    model = ConvVAE(params).to(device)

    # set the learning parameters
    lr = params["learning_rate"]  # 0.001
    epochs = 32
    batch_size = 128
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction="sum")
    # a list to save all the reconstructed images in PyTorch grid format
    grid_images = []  # noqa

    transform = transforms.Compose(
        [
            # transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )
    # training set and train data loader
    trainset = torchvision.datasets.MNIST(
        root="../input", train=True, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # validation set and validation data loader
    testset = torchvision.datasets.MNIST(
        root="../input", train=False, download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    train_loss = []
    valid_loss = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss = train(
            model, trainloader, trainset, device, optimizer, criterion
        )
        valid_epoch_loss, recon_images, rec_loss = validate(
            model, testloader, testset, device, criterion
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Loss: {valid_epoch_loss:.4f}")
        print(f"Rec Loss: {rec_loss:.4f}")

    print("TRAINING COMPLETE")

    # re-use old seeds
    np.random.set_state(old_np_seed)
    torch.set_rng_state(old_torch_seed)
    return min(valid_loss)


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
