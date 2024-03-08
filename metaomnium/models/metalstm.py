import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .learner import Learner
from .modules.utils import accuracy, get_loss_and_grads, miou, put_on_device,degree_loss, dist_acc, regression_loss, get_flat_params
from .modules.metalearner import MetaLearner


def compare_models(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.equal(p1, p2):
            return False
    return True


class MetaLSTM(Learner):

    def __init__(self, meta_batch_size=1, grad_clip=None, **kwargs):
        super().__init__(**kwargs)
        self.meta_batch_size = meta_batch_size
        self.grad_clip = grad_clip

        self.task_counter = 0
        self.learner_w_grad = self.baselearner_fn(**self.baselearner_args).to(
            self.dev)
        self.val_learner = self.baselearner_fn(**self.baselearner_args).to(
            self.dev)
        # save initial weights
        self.initialization = self.learner_w_grad.state_dict()     
        self.learner_wo_grad = copy.deepcopy(self.learner_w_grad)
        self.metalearner = MetaLearner(input_size=4, hidden_size=20, n_learner_params=get_flat_params(self.learner_w_grad).size(0)).to(self.dev)
        self.metalearner.metalstm.init_cI(get_flat_params(self.learner_w_grad))

        self.val_metalearner = MetaLearner(input_size=4, hidden_size=20, n_learner_params=get_flat_params(self.learner_w_grad).size(0)).to(self.dev)
        self.val_metalearner.metalstm.init_cI(get_flat_params(self.val_learner))

        self.initialization = copy.deepcopy(self.learner_w_grad)

        self.metalearner_init = self.metalearner.state_dict()
        self.val_metalearner_init = self.val_metalearner.state_dict()

        self.orig_cI = self.metalearner.metalstm.cI.data.clone()

        if self.opt_fn == "sgd":
            self.optimizer = torch.optim.SGD(
                self.metalearner.parameters(), lr=self.lr, momentum=self.momentum)
        else:
            self.optimizer = torch.optim.Adam(self.metalearner.parameters(), lr=self.lr)

    def _copy_flat_params_learner(self, model, cI):
        idx = 0
        for p in model.parameters():
            plen = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+plen].view_as(p))
            idx += plen

    def _preprocess_grad_loss(self, x):
        p = 10
        indicator = (x.abs() >= np.exp(-p)).to(torch.float32)

        # preproc1
        x_proc1 = indicator * torch.log(x.abs() + 1e-8) / p + (1 - indicator) * -1
        # preproc2
        x_proc2 = indicator * torch.sign(x) + (1 - indicator) * np.exp(p) * x
        return torch.stack((x_proc1, x_proc2), 1)

    def _train_learner(self, learner_w_grad, metalearner, T, train_x, train_y, task_type, evaluation=False):
        loss_history = list()
        cI = metalearner.metalstm.cI.data
        hs = [None]
        for _ in range(T):
            xinp, yinp = train_x, train_y
            self._copy_flat_params_learner(learner_w_grad, cI)
            loss, grads = get_loss_and_grads(learner_w_grad, xinp, yinp,
                                             weights=None, flat=False, task_type=task_type, item_loss=False,
                                             allow_unused_grad=True)
            loss_history.append(loss.item())

            # preprocess grad & loss and metalearner forward
            # flatten grads
            grads_flat = torch.cat([p.data.view(-1) for p in grads])
            grad_prep = self._preprocess_grad_loss(grads_flat)  # [n_learner_params, 2]
            loss_prep = self._preprocess_grad_loss(loss.data.unsqueeze(0)) # [1, 2]
            metalearner_input = [loss_prep, grad_prep, grads_flat.unsqueeze(1)]

            if evaluation:
                with torch.no_grad():
                    cI, h = metalearner(metalearner_input, hs[-1])
            else:
                cI, h = metalearner(metalearner_input, hs[-1])
            hs.append(h)

        return loss_history, cI

    def _deploy(self, learner_w_grad, metalearner, train_x, train_y, test_x, test_y, T, task_type, evaluation=False):
        # Train learner using meta-learner as optimizator
        loss_history, cI = self._train_learner(learner_w_grad, metalearner, T, train_x, train_y, task_type, evaluation=evaluation)

        # Train meta-learner with validation loss
        xinp, yinp = test_x, test_y
        self.learner_wo_grad.transfer_params(learner_w_grad, cI)
        out = self.learner_wo_grad(xinp, task_type=task_type)
        # out = learner.forward_weights(xinp, fast_weights, task_type=task_type)
        if task_type.startswith('regression'):
            test_loss = regression_loss(out, yinp, task_type, mode='train')
        else:
            test_loss = self.learner_wo_grad.criterion(out, yinp)
        loss_history.append(test_loss.item())
        with torch.no_grad():
            probs = F.softmax(out, dim=1)
            preds = torch.argmax(probs, dim=1)
            if task_type == "segmentation":
                score = miou(preds, test_y)
            elif task_type.startswith('regression'):
                score = regression_loss(out, test_y, task_type, mode='eval')
            else:
                score = accuracy(preds, test_y)
        
        return score, test_loss, loss_history, probs.cpu().numpy(), \
            preds.cpu().numpy()

    def train(self, train_x, train_y, test_x, test_y, task_type):
        self.learner_w_grad.reset_batch_stats()
        self.learner_wo_grad.reset_batch_stats()
        self.learner_w_grad.train()
        self.learner_wo_grad.train()
        self.metalearner.train()

        train_x, train_y, test_x, test_y = put_on_device(self.dev, [train_x,
                                                                    train_y, test_x, test_y])

        score, test_loss, _, probs, preds = self._deploy(self.learner_w_grad, self.metalearner,
                                                            train_x, train_y, test_x, test_y, self.T, task_type=task_type)
        
        self.optimizer.zero_grad()
        test_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.metalearner.parameters(), self.grad_clip)
        self.optimizer.step()

        # self.learner_w_grad.load_state_dict(self.learner_wo_grad.state_dict())

        return score, test_loss.item(), probs, preds
    
    def evaluate(self, num_classes, train_x, train_y, test_x, test_y, 
                 val=None, task_type=None):
        if num_classes is None:
            self.learner_w_grad.reset_batch_stats()
            self.learner_wo_grad.reset_batch_stats()
            learner = self.learner_w_grad
            metalearner = self.metalearner
            metalearner.metalstm.init_cI(get_flat_params(self.learner_w_grad))
        else:
            self.learner_w_grad.reset_batch_stats()
            self.learner_wo_grad.reset_batch_stats()
            self.val_learner.load_params(self.learner_w_grad.state_dict())
            self.val_learner.eval()            
            if task_type != "classification":
                self.val_learner.modify_out_layer(self.val_learner.model.out.out_features)
            else:
                self.val_learner.modify_out_layer(num_classes)

            self.val_metalearner.load_state_dict(self.metalearner.state_dict())
            self.val_metalearner.metalstm.init_cI(get_flat_params(self.val_learner))
            metalearner = self.val_metalearner
            learner = self.val_learner
            metalearner.eval()

        self.learner_wo_grad.eval()
        learner.train()

        train_x, train_y, test_x, test_y = put_on_device(self.dev,
                                                         [train_x, train_y, test_x, test_y])
        if val:
            T = self.T_val
        else:
            T = self.T_test

        score, _, loss_history, probs, preds = self._deploy(learner, metalearner,
                                                                train_x, train_y, test_x, test_y, T, task_type=task_type,
                                                                evaluation=True)
        
        return score, loss_history, probs, preds

    def dump_state(self):
        return {k: v.clone() for k, v in self.metalearner.state_dict().items()}

    def load_state(self, state):
        self.metalearner.eval()
        self.metalearner.load_state_dict(state)