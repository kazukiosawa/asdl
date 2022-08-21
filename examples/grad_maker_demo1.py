import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from asdfghjkl import GradientMaker


class NetworkA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 4)
        self.fc2 = nn.Linear(4, 3)

    def forward(self, inputs, targets=None, flip=False):
        h = self.fc1(inputs)
        logits = self.fc2(h)
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            if flip:
                return loss, loss  # returns a tuple of (loss, logits)
            return logits, loss  # returns a tuple of (logits, loss)
        return logits  # returns only logits


bs = 2
x = torch.randn(bs, 5)
t = torch.tensor([0] * bs, dtype=torch.long)

###########################################################
# Example 1: the model returns a tuple of logits and loss
###########################################################
model1 = NetworkA()
model2 = copy.deepcopy(model1)
grad_maker = GradientMaker(model2)

model1.zero_grad()
model2.zero_grad()

# standard
logits1, loss1 = model1(x, targets=t)
loss1.backward()

# GradientMaker
grad_maker.setup_model_call(model2, x, targets=t)
logits2, loss2 = grad_maker.forward_and_backward()

g1 = parameters_to_vector([p.grad for p in model1.parameters()])
g2 = parameters_to_vector([p.grad for p in model2.parameters()])
torch.testing.assert_close(logits1, logits2)
torch.testing.assert_close(loss1, loss2)
torch.testing.assert_close(g1, g2)


###########################################################
# Example 2: the model returns only logits
###########################################################
model1 = NetworkA()
model2 = copy.deepcopy(model1)
grad_maker = GradientMaker(model2)

model1.zero_grad()
model2.zero_grad()

# standard
logits1 = model1(x)
loss1 = F.cross_entropy(logits1, t)
loss1.backward()

# GradientMaker
logits = grad_maker.setup_model_call(model2, x)
grad_maker.setup_loss_call(F.cross_entropy, logits, t)
logits2, loss2 = grad_maker.forward_and_backward()

g1 = parameters_to_vector([p.grad for p in model1.parameters()])
g2 = parameters_to_vector([p.grad for p in model2.parameters()])
torch.testing.assert_close(logits1, logits2)
torch.testing.assert_close(loss1, loss2)
torch.testing.assert_close(g1, g2)


###########################################################
# Example 3: the model returns a tuple of loss and logits
###########################################################
model1 = NetworkA()
model2 = copy.deepcopy(model1)
grad_maker = GradientMaker(model2)

model1.zero_grad()
model2.zero_grad()

# standard
y = model1(x, targets=t, flip=True)
logits1 = y[1]
loss1 = y[0]
loss1.backward()

# GradientMaker
y = grad_maker.setup_model_call(model2, x, targets=t, flip=True)
grad_maker.setup_logits_repr(y[1])
grad_maker.setup_loss_repr(y[0])
logits2, loss2 = grad_maker.forward_and_backward()

g1 = parameters_to_vector([p.grad for p in model1.parameters()])
g2 = parameters_to_vector([p.grad for p in model2.parameters()])
torch.testing.assert_close(logits1, logits2)
torch.testing.assert_close(loss1, loss2)
torch.testing.assert_close(g1, g2)
