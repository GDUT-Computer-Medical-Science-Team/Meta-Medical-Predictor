import torch
from torch import nn
import torch.nn.functional as F


class RegressionModel(nn.Module):
    def __init__(self, input_size, n_hidden, output_size=1, dropoutRate=0.05):
        super(RegressionModel, self).__init__()
        # self.layer1 = nn.Linear(input_size, n_hidden)
        # self.layer2 = nn.Linear(n_hidden, int(n_hidden / 2))
        # self.layer3 = nn.Linear(int(n_hidden / 2), int(n_hidden / 4))
        # self.predict = nn.Linear(int(n_hidden / 4), output_size)

        # self.layer1 = nn.Conv1d(input_size, n_hidden, 1)
        # self.layer2 = nn.Conv1d(n_hidden, n_hidden, 1)
        # self.layer3 = nn.Conv1d(n_hidden, n_hidden, 1)

        self.layer1 = nn.Linear(input_size, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_hidden)
        self.layer3 = nn.Linear(n_hidden, n_hidden)
        # self.layer4 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, output_size)

        self.activate = nn.Tanh()
        self.dropoutRate = dropoutRate

    def forward(self, x):
        # x = x.unsqueeze(0)
        out = self.layer1(x)
        out = self.activate(out)
        out = F.dropout(out, p=self.dropoutRate)

        out = self.layer2(out)
        out = self.activate(out)
        out = F.dropout(out, p=self.dropoutRate)

        out = self.layer3(out)
        out = self.activate(out)
        out = F.dropout(out, p=self.dropoutRate)

        # out = self.layer4(out)
        # out = self.activate(out)
        # out = F.dropout(out, p=self.dropoutRate)

        out = torch.sigmoid(out)
        # out = torch.tanh(out)
        out = self.predict(out)
        return out
#
# class RegressionModel(torch.nn.Module):
#     def __init__(self):
#         super(RegressionModel, self).__init__()
#         self.hidden_size = 32
#         self.fc1 = torch.nn.Linear(1, self.hidden_size)
#         self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
#         self.fc3 = torch.nn.Linear(self.hidden_size, 1)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#
# class MetaRegressionModel(torch.nn.Module):
#     def __init__(self, model=None):
#         super(MetaRegressionModel, self).__init__()
#         if model is None:
#             self.model = RegressionModel()
#         else:
#             self.model = model
#
#     def forward(self, inputs, params=None):
#         if params is None:
#             params = self.model.parameters()
#         x, y = inputs
#         y_hat = self.model(x, params=params)
#         loss = F.mse_loss(y_hat, y)
#         return loss, y_hat
#
#     def adapt(self, train_inputs, train_outputs):
#         self.model.train()
#         self.meta_optim.zero_grad()
#         train_loss, _ = self.forward((train_inputs, train_outputs))
#         train_loss.backward()
#         self.meta_optim.step()
#
#     def meta_update(self, train_inputs, train_outputs, test_inputs, test_outputs):
#         self.model.train()
#         params = l2l.clone_parameters(self.model.parameters())
#         train_loss, _ = self.forward((train_inputs, train_outputs))
#         grads = torch.autograd.grad(train_loss, params)
#         fast_weights = l2l.update_parameters(self.model.parameters(), grads, step_size=self.step_size)
#         test_loss, y_hat = self.forward((test_inputs, test_outputs), params=fast_weights)
#         return test_loss, y_hat
#
#     def meta_test(self, test_inputs):
#         self.model.eval()
#         y_hat = self.model(test_inputs)
#         return y_hat
