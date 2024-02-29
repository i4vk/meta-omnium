import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .learner import Learner
from .modules.utils import accuracy, get_loss_and_grads, miou, put_on_device,degree_loss, dist_acc, regression_loss, get_flat_params
from .modules.metalearner import MetaLearner
class MetaLSTM(Learner):

    def __init__(self, meta_lr, meta_batch_size=1, grad_clip=None, **kwargs):
        super().__init__(**kwargs)
        self.meta_lr = meta_lr
        self.meta_batch_size = meta_batch_size
        self.grad_clip = grad_clip

        self.task_counter = 0
        self.learner_w_grad = self.baselearner_fn(**self.baselearner_args).to(
            self.dev)
        self.val_learner = self.baselearner_fn(**self.baselearner_args).to(
            self.dev)
        self.initialization = [p.clone().detach().to(self.dev) for p in
                               self.learner_w_grad.parameters()]
        self.learner_wo_grad = copy.deepcopy(self.learner_w_grad)
        self.metalearner = MetaLearner(input_size=4, hidden_size=20, n_learner_params=get_flat_params(self.learner_w_grad).size(0)).to(self.dev)
        self.metalearner.metalstm.init_cI(get_flat_params(self.learner_w_grad))
        self.metalearner_initialization = [p.clone().detach().to(self.dev) for p in
                                             self.metalearner.parameters()]

        # initialization tiene que ser sobre los parámetros del learner o del metalearner¿?
        for p in self.initialization:
            p.requires_grad = True
        for p in self.metalearner_initialization:
            p.requires_grad = True

        if self.opt_fn == "sgd":
            self.optimizer = torch.optim.SGD(
                self.metalearner.parameters(), lr=self.meta_lr, momentum=self.momentum)
        else:
            self.optimizer = torch.optim.Adam(self.metalearner.parameters(), lr=self.meta_lr)

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

    def _train_learner(self, learner_w_grad, metalearner, T, train_x, train_y, fast_weights, task_type):
        loss_history = list()
        cI = metalearner.metalstm.cI.data
        hs = [None]
        for _ in range(T):
            xinp, yinp = train_x, train_y
            self._copy_flat_params_learner(learner_w_grad, cI)
            loss, grads = get_loss_and_grads(learner_w_grad, xinp, yinp,
                                             weights=fast_weights, flat=False, task_type=task_type, item_loss=False)
            loss_history.append(loss)

            # preprocess grad & loss and metalearner forward
            # flatten grads
            grads_flat = torch.cat([p.data.view(-1) for p in grads])
            grad_prep = self._preprocess_grad_loss(grads_flat)  # [n_learner_params, 2]
            loss_prep = self._preprocess_grad_loss(loss.data.unsqueeze(0)) # [1, 2]
            metalearner_input = [loss_prep, grad_prep, grads_flat.unsqueeze(1)]
            cI, h = metalearner(metalearner_input, hs[-1])
            hs.append(h)

        return loss_history, cI

    def _deploy(self, learner, metalearner, train_x, train_y, test_x, test_y, T, fast_weights, task_type):
        # Train learner using meta-learner as optimizator
        loss_history, cI = self._train_learner(learner, metalearner, T, train_x, train_y, fast_weights, task_type)

        # Train meta-learner with validation loss
        xinp, yinp = test_x, test_y
        self.learner_wo_grad.transfer_params(learner, cI)
        out = learner.forward_weights(xinp, fast_weights, task_type=task_type)
        if task_type.startswith('regression'):
            test_loss = regression_loss(out, yinp, task_type, mode='train')
        else:
            test_loss = learner.criterion(out, yinp)
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

        train_x, train_y, test_x, test_y = put_on_device(self.dev, [train_x,
                                                                    train_y, test_x, test_y])
        
        fast_weights = [p.clone() for p in self.initialization]
        # Get the weights only used for the given task
        if task_type == "classification":
            filtered_fast_weights = fast_weights[:-16]
        elif task_type == "segmentation":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-16:-14]
        elif task_type == "regression_pose_animals":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-14:-12]
        elif task_type == 'regression_pose_animals_syn':
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-12:-10]
        elif task_type == "regression_mpii":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-10:-8]
        elif task_type == "regression_distractor":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-8:-6]
        elif task_type == "regression_pascal1d":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-6:-4]
        elif task_type == "regression_shapenet1d":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-4:-2]
        elif task_type == "regression_shapenet2d":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-2:]

        score, test_loss, _, probs, preds = self._deploy(self.learner_w_grad, self.metalearner,
                                                         train_x, train_y, test_x, test_y, self.T, filtered_fast_weights, task_type)
        
        self.optimizer.zero_grad()
        test_loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.metalearner.parameters(), self.grad_clip)
        self.optimizer.step()

        return score, test_loss.item(), probs, preds
    
    def evaluate(self, num_classes, train_x, train_y, test_x, test_y, 
                 val=None, task_type=None):
        if num_classes is None:
            self.learner_w_grad.reset_batch_stats()
            self.learner_wo_grad.reset_batch_stats()
            self.learner_w_grad.eval()
            self.learner_wo_grad.eval()
            fast_weights = [p.clone() for p in self.initialization]
            metalearner = self.metalearner
            metalearner.metalstm.init_cI(get_flat_params(self.learner_w_grad))
        else:
            self.val_learner.load_params(self.learner_w_grad.state_dict())
            self.val_learner.eval()
            self.val_learner.modify_out_layer(num_classes)

            fast_weights = [p.clone() for p in self.initialization[:-18]]

            initialization = [p.clone().detach().to(self.dev) for p in 
                              self.val_learner.parameters()]
            
            fast_weights.extend(initialization[-18:])
            for p in fast_weights[-18:]:
                p.requires_grad = True
            metalearner = self.metalearner
            metalearner.metalstm.init_cI(get_flat_params(self.val_learner))
            learner = self.val_learner

        if task_type == "classification":
            filtered_fast_weights = fast_weights[:-16]
        elif task_type == "segmentation":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-16:-14]
        elif task_type == "regression_pose_animals":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-14:-12]
        elif task_type == 'regression_pose_animals_syn':
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-12:-10]
        elif task_type == "regression_mpii":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-10:-8]
        elif task_type == "regression_distractor":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-8:-6]
        elif task_type == "regression_pascal1d":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-6:-4]
        elif task_type == "regression_shapenet1d":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-4:-2]
        elif task_type == "regression_shapenet2d":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-2:]

        train_x, train_y, test_x, test_y = put_on_device(self.dev,
                                                         [train_x, train_y, test_x, test_y])
        if val:
            T = self.T_val
        else:
            T = self.T_test

        score, _, loss_history, probs, preds = self._deploy(learner, metalearner,
                                                            train_x, train_y, test_x, test_y, T, filtered_fast_weights, task_type=task_type)
        # score, _, loss_history, probs, preds = self._deploy(learner, metalearner,
        #                                                     train_x, train_y, test_x, test_y, T, fast_weights, task_type=task_type)
        
        return score, loss_history, probs, preds            
    
    def dump_state(self):
        return [p.clone().detach() for p in self.initialization], [p.clone().detach() for p in self.metalearner_initialization]

    def load_state(self, state, metalearner_state):
        self.initialization = [p.clone() for p in state]
        for p in self.initialization:
            p.requires_grad = True            
        self.metalearner_initialization = [p.clone() for p in metalearner_state]
        for p in self.metalearner_initialization:
            p.requires_grad = True