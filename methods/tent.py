import csv
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import numpy as np
from tqdm import tqdm

class TentTrainer:

    def __init__(self, params, setting):

        self.device = params['device']

        self.train_loader = params['train_loader']
        self.val_loader = params['val_loader']

        self.model = params['model']
        self.model = configure_model(self.model)
        model_params, _ = collect_params(self.model)
        self.optimizer = params['optimizer']['type'](model_params, lr=params['optimizer']['lr'])
        self.tented_model = Tent(self.model, self.optimizer)

        self.setting = setting

        self.all_cA = []
        self.all_cB = []

    def train_epoch(self, epoch):

        losses = []

        for B, _ in tqdm(self.train_loader['B'], total=len(self.train_loader['B'])):

            B = B.to(self.device)
            loss = self.tented_model(B)
            losses.append(loss)

        print('----- Epoch', epoch, '-----')
        print('Entropy:', np.mean(losses))
        print('------------------------')
        print()

    def evaluate(self, epoch):

        class_preds = { 'A' : [], 'B' : [] }
        class_labels = { 'A' : [], 'B' : [] }

        for i, ((A, A_labels), (B, B_labels)) in enumerate(zip(self.val_loader['A'], self.val_loader['B'])):

            A, B = A.to(self.device), B.to(self.device)

            class_preds['A'] += self.tented_model(A, train=False).cpu().detach().argmax(dim=1).tolist()
            class_labels['A'] += A_labels.tolist()

            class_preds['B'] += self.tented_model(B, train=False).cpu().detach().argmax(dim=1).tolist()
            class_labels['B'] += B_labels.tolist()

        self.all_cA.append(np.mean(np.array(class_preds['A']) == np.array(class_labels['A'])))
        self.all_cB.append(np.mean(np.array(class_preds['B']) == np.array(class_labels['B'])))

        print('===== Classification =====')
        print('(A) acc:', self.all_cA[-1])
        print('(B) acc:', self.all_cB[-1])
        print('==========================')
        print()

    def train(self, n_epochs=500, epoch_step=1):

        for epoch in range(1, n_epochs+1):
            self.train_epoch(epoch)

            if epoch % epoch_step == 0:
                self.evaluate(epoch)

    def write_results(self):

        with open('results/metrics.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([self.setting, self.all_cA[-1], self.all_cB[-1],'', '', self.all_cA,
                            self.all_cB, '', ''])

    def write_model(self):
        torch.save(self.tented_model.state_dict(), 'models/' + self.setting + '.pt')

class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, train=True):
        if self.episodic:
            self.reset()

        if train:
            self.model.train()
            for _ in range(self.steps):
                loss = forward_and_adapt(x, self.model, self.optimizer)
            return loss
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
            return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """

    # forward
    outputs = model(x)
    # adapt
    loss = softmax_entropy(outputs).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return float(loss)


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
