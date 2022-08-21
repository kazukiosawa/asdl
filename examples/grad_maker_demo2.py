import copy
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from asdfghjkl import GradientMaker


@dataclass
class Output:
    loss: torch.Tensor
    logits: torch.Tensor
    hidden_states: torch.Tensor


class NetworkB(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 4)
        self.fc2 = nn.Linear(4, 3)

    def forward(self, inputs, targets=None, return_dict=True):
        h = self.fc1(inputs)
        logits = self.fc2(h)
        loss = None
        if targets is not None:
            # flatten (bs, seq) dimensions
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        # Hugging Face Transformers' style
        if not return_dict:
            output = (logits, h)
            return ((loss,) + output) if loss is not None else output
        else:
            return Output(loss=loss, logits=logits, hidden_states=h)


bs = 2
seq = 3
x = torch.randn(bs, seq, 5)  # sequence modeling style inputs
t = torch.zeros(bs, seq, dtype=torch.long)  # sequence modeling style targets

###########################################################
# Example 1: the model returns Output w/o loss
###########################################################
model1 = NetworkB()
model2 = copy.deepcopy(model1)
grad_maker = GradientMaker(model2)

model1.zero_grad()
model2.zero_grad()

# standard
y = model1(x)
logits1 = y.logits
loss1 = F.cross_entropy(logits1.view(-1, logits1.size(-1)), t.view(-1), ignore_index=-1)
loss1.backward()

# GradientMaker
y = grad_maker.setup_model_call(model2, x)
grad_maker.setup_logits_repr(y.logits)
grad_maker.setup_loss_call(F.cross_entropy, y.logits.view(-1, y.logits.size(-1)), t.view(-1), ignore_index=-1)
logits2, loss2 = grad_maker.forward_and_backward()

g1 = parameters_to_vector([p.grad for p in model1.parameters()])
g2 = parameters_to_vector([p.grad for p in model2.parameters()])
torch.testing.assert_close(logits1, logits2)
torch.testing.assert_close(loss1, loss2)
torch.testing.assert_close(g1, g2)


###########################################################
# Example 2: the model returns tuple w/ loss
###########################################################
model1 = NetworkB()
model2 = copy.deepcopy(model1)
grad_maker = GradientMaker(model2)

model1.zero_grad()
model2.zero_grad()

# standard
y = model1(x, targets=t, return_dict=False)
logits1 = y[1]
loss1 = y[0]
loss1.backward()

# GradientMaker
y = grad_maker.setup_model_call(model2, x, targets=t, return_dict=False)
grad_maker.setup_logits_repr(y[1])
grad_maker.setup_loss_repr(y[0])
logits2, loss2 = grad_maker.forward_and_backward()

g1 = parameters_to_vector([p.grad for p in model1.parameters()])
g2 = parameters_to_vector([p.grad for p in model2.parameters()])
torch.testing.assert_close(logits1, logits2)
torch.testing.assert_close(loss1, loss2)
torch.testing.assert_close(g1, g2)
