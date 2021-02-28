import torch
from torch.utils import data
from utils.buffer.buffer import Buffer
from agents.base import ContinualLearner
from continuum.data_utils import dataset_transform
from utils.setup_elements import transforms_match, input_size_match
from utils.utils import maybe_cuda, AverageMeter
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn

class SupContrastPure(ContinualLearner):
    def __init__(self, model, opt, params):
        super(SupContrastPure, self).__init__(model, opt, params)
        self.buffer = Buffer(model, params)
        params.mem_size = int(params.mem_size*0.2)
        params.eps_mem_batch =params.batch
        params.retrieve = 'match'
        self.aux_params = params
        self.mem_size = params.mem_size
        self.eps_mem_batch = params.eps_mem_batch
        self.mem_iters = params.mem_iters


    def train_learner(self, x_train, y_train):

        self.aux_buffer = Buffer(self.model, self.aux_params)
        self.before_train(x_train, y_train)
        # set up loader
        train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[self.data])
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch, shuffle=True, num_workers=0,
                                       drop_last=True)
        # set up model
        self.model = self.model.train()

        # setup tracker
        losses = AverageMeter()
        acc_batch = AverageMeter()


        for ep in range(self.epoch):
            for i, batch_data in enumerate(train_loader):
                # batch update
                batch_x, batch_y = batch_data
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                batch_aux_x, batch_aux_y = self.aux_buffer.retrieve(x=batch_x, y=batch_y)
                batch_aux_x = maybe_cuda(batch_aux_x, self.cuda)
                batch_aux_y = maybe_cuda(batch_aux_y, self.cuda)
                if batch_aux_y.size(0) > 0:
                    for j in range(self.mem_iters):
                        candidate_x, candidate_y, match_x, match_y = self.buffer.retrieve()
                        if candidate_x.size(0) > 0:
                            candidate_x = maybe_cuda(candidate_x, self.cuda)
                            candidate_y = maybe_cuda(candidate_y, self.cuda)
                            match_x = maybe_cuda(match_x, self.cuda)
                            match_y = maybe_cuda(match_y, self.cuda)

                            combined_batch_1 = torch.cat((match_x, batch_x))
                            combined_labels_1 = torch.cat((match_y, batch_y))

                            combined_batch_2= torch.cat((candidate_x, batch_aux_x))
                            combined_labels_2 = torch.cat((candidate_y, batch_aux_y))

                            features = torch.cat([self.model.forward(combined_batch_1), self.model.forward(combined_batch_2)], dim=0).unsqueeze(1)
                            combined_labels = torch.cat([combined_labels_1, combined_labels_2], dim=0)
                            loss = self.criterion(features, combined_labels)
                            losses.update(loss, batch_y.size(0))
                            self.opt.zero_grad()
                            loss.backward()
                            self.opt.step()
                # update mem
                if self.buffer.current_index < self.buffer.buffer_label.size(0):
                    self.buffer.update(batch_x, batch_y)
                self.aux_buffer.update(batch_x, batch_y)
                if i % 100 == 1 and self.verbose:
                        print(
                            '==>>> it: {}, avg. loss: {:.6f}, '
                                .format(i, losses.avg(), acc_batch.avg())
                        )
        self.buffer.update(self.aux_buffer.buffer_img, self.aux_buffer.buffer_label)
        self.after_train()
