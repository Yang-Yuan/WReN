import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.basic_model import BasicModel


class conv_module(nn.Module):
    def __init__(self):
        super(conv_module, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()
        # self.fc = nn.Linear(32*4*4, 256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.batch_norm1(x))
        x = self.conv2(x)
        x = self.relu2(self.batch_norm2(x))
        x = self.conv3(x)
        x = self.relu3(self.batch_norm3(x))
        x = self.conv4(x)
        x = self.relu4(self.batch_norm4(x))
        return x


class relation_module(nn.Module):
    def __init__(self):
        super(relation_module, self).__init__()
        self.fc1 = nn.Linear(256*2, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 512)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(512, 256)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        return x


class mlp_module(nn.Module):
    def __init__(self):
        super(mlp_module, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 13)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.view(-1, 8, 13)

# class panels_to_embeddings(nn.Module):
#     def __init__(self, tag):
#         super(panels_to_embeddings, self).__init__()
#         self.in_dim = 512
#         if tag:
#             self.in_dim += 9
#         self.fc = nn.Linear(self.in_dim, 256)
#
#     def forward(self, x):
#         return self.fc(x.view(-1, self.in_dim))


class combinatorial(nn.Module):
    def __init__(self, batch_size):
        super(combinatorial, self).__init__()
        self.batch_size = batch_size

    def forward(self, features):
        # output size should be either [batch_size, choice_num, pair_num = 9*9 or 9*8]
        # but the original method gives 80 at the third dimension
        # since position tags are included in features,
        # the identical entries can be distinguished,
        # we can use 9*9 pairs

        context = features[:, :8, :]
        choice = features[:, 8:, :]

        # duplicate context matrix as many times as the number of answer choices
        # complete the matrices with each choice
        context = context.unsqueeze(1).expand(-1, 8, -1, -1)
        choice = choice.unsqueeze(2)
        features = torch.cat((context, choice), dim = 2)

        # take pairs of matrix entries
        features = torch.cat((features.unsqueeze(2).expand(-1, -1, 9, -1, -1),
                              features.unsqueeze(3).expand(-1, -1, -1, 9, -1)), dim = -1)
        features = features.view(self.batch_size, 8, 9 * 9, 512)
        return features

        # context_embeddings_pairs = torch.cat((context_embeddings.unsqueeze(1).expand(-1, 8, -1, -1),
        #                                       context_embeddings.unsqueeze(2).expand(-1, -1, 8, -1)),
        #                                      dim=3).view(-1, 64, 512)
        # context_embeddings = context_embeddings.unsqueeze(1).expand(-1, 8, -1, -1)
        # choice_embeddings = choice_embeddings.unsqueeze(2).expand(-1, -1, 8, -1)
        # choice_context_order = torch.cat((context_embeddings, choice_embeddings), dim=3)
        # choice_context_reverse = torch.cat((choice_embeddings, context_embeddings), dim=3)
        # embedding_paris = [context_embeddings_pairs.unsqueeze(1).expand(-1, 8, -1, -1), choice_context_order, choice_context_reverse]
        # return torch.cat(embedding_paris, dim=2).view(-1, 8, 80, 512)


class WReN(BasicModel):
    def __init__(self, args):
        super(WReN, self).__init__(args)

        self.img_size = args.img_size
        self.meta_beta = args.meta_beta
        self.use_tag = args.tag
        self.use_cuda = args.cuda
        self.batch_size = args.batch_size
        if self.use_tag:
            self.tags = self.build_tags()

        self.conv = conv_module()
        self.lin_proj = nn.Linear(512 + 9, 256) if self.use_tag else nn.Linear(512, 256)
        self.comb = combinatorial(self.batch_size)
        self.rn = relation_module()
        self.mlp = mlp_module()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)

    def build_tags(self):
        tags = np.zeros((16, 9))
        tags[:8, :8] = np.eye(8)
        tags[8:, 8] = 1
        tags = torch.tensor(tags, dtype = torch.float)
        tags = tags.unsqueeze(0).expand(self.batch_size, -1, -1)
        tags = tags.cuda()
        return tags

    # def tag_panels(self, batch_size):
    #     tags = []
    #     for idx in range(0, 16):
    #         tag = np.zeros([1, 9], dtype=float)
    #         if idx < 8:
    #             tag[:, idx] = 1.0
    #         else:
    #             tag[:, 8] = 1.0
    #         tag = torch.tensor(tag, dtype=torch.float).expand(batch_size, -1).unsqueeze(1)
    #         if self.use_cuda:
    #             tag = tag.cuda()
    #         tags.append(tag)
    #     tags = torch.cat(tags, dim=1)
    #     return tags

    # def group_panel_embeddings(self, embeddings):
    #     embeddings = embeddings.view(-1, 16, 256)
    #     embeddings_seq = torch.chunk(embeddings, 16, dim=1)
    #     context_pairs = []
    #     for context_idx1 in range(0, 8):
    #         for context_idx2 in range(0, 8):
    #             if not context_idx1 == context_idx2:
    #                 context_pairs.append(torch.cat((embeddings_seq[context_idx1], embeddings_seq[context_idx2]), dim=2))
    #     context_pairs = torch.cat(context_pairs, dim=1)
    #     panel_embeddings_pairs = []
    #     for answer_idx in range(8, len(embeddings_seq)):
    #         embeddings_pairs = context_pairs
    #         for context_idx in range(0, 8):
    #             # In order
    #             order = torch.cat((embeddings_seq[answer_idx], embeddings_seq[context_idx]), dim=2)
    #             reverse = torch.cat((embeddings_seq[context_idx], embeddings_seq[answer_idx]), dim=2)
    #             choice_pairs = torch.cat((order, reverse), dim=1)
    #             embeddings_pairs = torch.cat((embeddings_pairs, choice_pairs), dim=1)
    #         panel_embeddings_pairs.append(embeddings_pairs.unsqueeze(1))
    #     panel_embeddings_pairs = torch.cat(panel_embeddings_pairs, dim=1)
    #     return panel_embeddings_pairs.view(-1, 8, 72, 512)


    # def rn_sum_features(self, features):
    #     features = features.view(-1, 8, 80, 256)
    #     sum_features = torch.sum(features, dim=2)
    #     return sum_features

    # def compute_loss(self, output, target, meta_target):
    #     pred, meta_pred = output[0], output[1]
    #     target_loss = F.cross_entropy(pred, target)
    #     meta_pred = torch.chunk(meta_pred, chunks=12, dim=1)
    #     meta_target = torch.chunk(meta_target, chunks=12, dim=1)
    #     meta_target_loss = 0.
    #     for idx in range(0, 12):
    #         meta_target_loss += F.binary_cross_entropy(F.sigmoid(meta_pred[idx]), meta_target[idx])
    #     loss = target_loss + self.meta_beta*meta_target_loss / 12.
    #     return loss

    def forward(self, x):
        # entry-wise encoding
        x = x.view(-1, 1, self.img_size, self.img_size)
        entry_features = self.conv(x)
        entry_features = entry_features.view(self.batch_size, 16, -1) # restore batch format
        if self.use_tag:
            entry_features = torch.cat((entry_features, self.tags), dim=2) # position tags if necessary

        # reduce feature size through linear
        entry_features = self.lin_proj(entry_features)

        # given the matrix completed by each answer choice
        # take all pairs of matrix entries
        entry_pair_features = self.comb(entry_features)

        # the linear can hand the input without reshape
        # entry_relational_features = self.rn(entry_pair_features.view(-1, 512))
        relational_features = self.rn(entry_pair_features)

        # sum_features = self.rn_sum_features(entry_relational_features)
        relational_features_aggregated = torch.sum(relational_features, dim = 2)

        # output = self.mlp(sum_features.view(-1, 256))
        output = self.mlp(relational_features_aggregated)

        pred = output[:, :, 12]

        # this summation is so in the original paper, but it does not seem very effective
        # because the meta-target should be predicted from the context whether it is
        # correctly completed or not
        meta_pred = torch.sum(output[:, :, 0:12], dim=1)

        return pred, meta_pred