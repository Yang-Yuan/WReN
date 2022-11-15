import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicModel(nn.Module):
    def __init__(self, args):
        super(BasicModel, self).__init__()
        self.name = args.model
        self.meta_beta = args.meta_beta

    def load_model(self, path, epoch):
        self.state_dict = torch.load(path+'{}_epoch_{}.pth'.format(self.name, epoch))

    def save_model(self, path, epoch):
        torch.save(self.state_dict(), path+'{}_epoch_{}.pth'.format(self.name, epoch))

    def compute_loss(self, pred, meta_pred, target, meta_target):
        target_loss = F.cross_entropy(pred, target)

        meta_target_loss = 0.
        if meta_pred is not None:
            # The BCE loss function can handle multi-dimension input
            # it is actually no difference between batch and feature size dimensions
            # just give the loss function all the things and it processes it element-wise
            # the aggregation part might be different to the following method
            # but as a loss, it is equivalent functionally
            meta_target_loss = F.binary_cross_entropy(torch.sigmoid(meta_pred), meta_target)

            # meta_pred = torch.chunk(meta_pred, chunks=12, dim=1)
            # meta_target = torch.chunk(meta_target, chunks=12, dim=1)
            # for idx in range(0, 12):
            #     meta_target_loss += F.binary_cross_entropy(torch.sigmoid(meta_pred[idx]), meta_target[idx])

        loss = target_loss + self.meta_beta*meta_target_loss / 12.
        return loss

    def train_(self, input, target, meta_target):

        self.optimizer.zero_grad()
        output = self(input)

        if isinstance(output, tuple):
            pred, meta_pred = output
        else:
            pred = output
            meta_pred = None

        loss = self.compute_loss(pred, meta_pred, target, meta_target)
        loss.backward()
        self.optimizer.step()

        pred = pred.data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100. / target.size()[0]
        return loss.item(), accuracy

    def validate_(self, input, target, meta_target):
        output = self(input)

        if isinstance(output, tuple):
            pred, meta_pred = output
        else:
            pred = output
            meta_pred = None

        loss = self.compute_loss(pred, meta_pred, target, meta_target)

        pred = pred.data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100. / target.size()[0]
        return loss.item(), accuracy

    def test_(self, input, target, meta_target):
        output = self(input)

        if isinstance(output, tuple):
            pred, meta_pred = output
        else:
            pred = output
            meta_pred = None

        pred = pred.data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()
        accuracy = correct * 100. / target.size()[0]
        return accuracy