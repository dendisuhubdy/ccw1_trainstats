import os
import time
import glob
import torch
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch import optim
from torch import nn
from torchtext import data
from torchtext import datasets
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from argparse import ArgumentParser
from visdom import Visdom

viz = Visdom(server='http://suhubdy.com', port=51401)

class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(*size, -1)


class Linear(Bottle, nn.Linear):
    pass


class BatchNorm(Bottle, nn.BatchNorm1d):
    pass


class Feature(nn.Module):

    def __init__(self, size, dropout):
        super(Feature, self).__init__()
        self.bn = nn.BatchNorm1d(size * 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, prem, hypo):
        return self.dropout(self.bn(torch.cat(
            [prem, hypo, prem - hypo, prem * hypo], 1)))


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        input_size = config.d_proj if config.projection else config.d_embed
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.d_hidden,
                           num_layers=config.n_layers, dropout=config.rnn_dropout,
                           bidirectional=config.birnn)

    def forward(self, inputs, _):
        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        h0 = c0 = Variable(inputs.data.new(*state_shape).zero_())
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return ht[-1] if not self.config.birnn else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)


class SNLIClassifier(nn.Module):

    def __init__(self, config):
        super(SNLIClassifier, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.projection = Linear(config.d_embed, config.d_proj)
        self.embed_bn = BatchNorm(config.d_proj)
        self.embed_dropout = nn.Dropout(p=config.embed_dropout)
        self.encoder = SPINN(config) if config.spinn else Encoder(config)
        feat_in_size = config.d_hidden * (
            2 if self.config.birnn and not self.config.spinn else 1)
        self.feature = Feature(feat_in_size, config.mlp_dropout)
        self.mlp_dropout = nn.Dropout(p=config.mlp_dropout)
        self.relu = nn.ReLU()
        mlp_in_size = 4 * feat_in_size
        mlp = [nn.Linear(mlp_in_size, config.d_mlp), self.relu,
               nn.BatchNorm1d(config.d_mlp), self.mlp_dropout]
        for i in range(config.n_mlp_layers - 1):
            mlp.extend([nn.Linear(config.d_mlp, config.d_mlp), self.relu,
                        nn.BatchNorm1d(config.d_mlp), self.mlp_dropout])
        mlp.append(nn.Linear(config.d_mlp, config.d_out))
        self.out = nn.Sequential(*mlp)

    def forward(self, batch):
        # import pdb
        # pdb.set_trace()
        prem_embed = self.embed(batch.premise)
        hypo_embed = self.embed(batch.hypothesis)
        if self.config.fix_emb:
            prem_embed = Variable(prem_embed.data)
            hypo_embed = Variable(hypo_embed.data)
        if self.config.projection:
            prem_embed = self.projection(prem_embed)  # no relu
            hypo_embed = self.projection(hypo_embed)
        prem_embed = self.embed_dropout(self.embed_bn(prem_embed))
        hypo_embed = self.embed_dropout(self.embed_bn(hypo_embed))
        if hasattr(batch, 'premise_transitions'):
            prem_trans = batch.premise_transitions
            hypo_trans = batch.hypothesis_transitions
        else:
            prem_trans = hypo_trans = None
        premise = self.encoder(prem_embed, prem_trans)
        hypothesis = self.encoder(hypo_embed, hypo_trans)
        scores = self.out(self.feature(premise, hypothesis))
        #print(premise[0][:5], hypothesis[0][:5])
        return scores

def tree_lstm(c1, c2, lstm_in):
    a, i, f1, f2, o = lstm_in.chunk(5, 1)
    c = a.tanh() * i.sigmoid() + f1.sigmoid() * c1 + f2.sigmoid() * c2
    h = o.sigmoid() * c.tanh()
    return h, c


def bundle(lstm_iter):
    if lstm_iter is None:
        return None
    lstm_iter = tuple(lstm_iter)
    if lstm_iter[0] is None:
        return None
    return torch.cat(lstm_iter, 0).chunk(2, 1)


def unbundle(state):
    if state is None:
        return itertools.repeat(None)
    return torch.split(torch.cat(state, 1), 1, 0)


class Reduce(nn.Module):
    """TreeLSTM composition module for SPINN.
    The TreeLSTM has two or three inputs: the first two are the left and right
    children being composed; the third is the current state of the tracker
    LSTM if one is present in the SPINN model.
    Args:
        size: The size of the model state.
        tracker_size: The size of the tracker LSTM hidden state, or None if no
            tracker is present.
    """

    def __init__(self, size, tracker_size=None):
        super(Reduce, self).__init__()
        self.left = nn.Linear(size, 5 * size)
        self.right = nn.Linear(size, 5 * size, bias=False)
        if tracker_size is not None:
            self.track = nn.Linear(tracker_size, 5 * size, bias=False)

    def forward(self, left_in, right_in, tracking=None):
        """Perform batched TreeLSTM composition.
        This implements the REDUCE operation of a SPINN in parallel for a
        batch of nodes. The batch size is flexible; only provide this function
        the nodes that actually need to be REDUCEd.
        The TreeLSTM has two or three inputs: the first two are the left and
        right children being composed; the third is the current state of the
        tracker LSTM if one is present in the SPINN model. All are provided as
        iterables and batched internally into tensors.
        Additionally augments each new node with pointers to its children.
        Args:
            left_in: Iterable of ``B`` ~autograd.Variable objects containing
                ``c`` and ``h`` concatenated for the left child of each node
                in the batch.
            right_in: Iterable of ``B`` ~autograd.Variable objects containing
                ``c`` and ``h`` concatenated for the right child of each node
                in the batch.
            tracking: Iterable of ``B`` ~autograd.Variable objects containing
                ``c`` and ``h`` concatenated for the tracker LSTM state of
                each node in the batch, or None.
        Returns:
            out: Tuple of ``B`` ~autograd.Variable objects containing ``c`` and
                ``h`` concatenated for the LSTM state of each new node. These
                objects are also augmented with ``left`` and ``right``
                attributes.
        """
        left, right = bundle(left_in), bundle(right_in)
        tracking = bundle(tracking)
        lstm_in = self.left(left[0])
        lstm_in += self.right(right[0])
        if hasattr(self, 'track'):
            lstm_in += self.track(tracking[0])
        out = unbundle(tree_lstm(left[1], right[1], lstm_in))
        # for o, l, r in zip(out, left_in, right_in):
        #     o.left, o.right = l, r
        return out


class Tracker(nn.Module):

    def __init__(self, size, tracker_size, predict):
        super(Tracker, self).__init__()
        self.rnn = nn.LSTMCell(3 * size, tracker_size)
        if predict:
            self.transition = nn.Linear(tracker_size, 4)
        self.state_size = tracker_size

    def reset_state(self):
        self.state = None

    def forward(self, bufs, stacks):
        buf = bundle(buf[-1] for buf in bufs)[0]
        stack1 = bundle(stack[-1] for stack in stacks)[0]
        stack2 = bundle(stack[-2] for stack in stacks)[0]
        x = torch.cat((buf, stack1, stack2), 1)
        if self.state is None:
            self.state = 2 * [Variable(
                x.data.new(x.size(0), self.state_size).zero_())] 
        self.state = self.rnn(x, self.state)
        if hasattr(self, 'transition'):
            return unbundle(self.state), self.transition(self.state[0])
        return unbundle(self.state), None

class SPINN(nn.Module):

    def __init__(self, config):
        super(SPINN, self).__init__()
        self.config = config
        assert config.d_hidden == config.d_proj / 2
        self.reduce = Reduce(config.d_hidden, config.d_tracker)
        if config.d_tracker is not None:
            self.tracker = Tracker(config.d_hidden, config.d_tracker,
                                   predict=config.predict)

    def forward(self, buffers, transitions):
        buffers = [list(torch.split(b.squeeze(1), 1, 0))
                   for b in torch.split(buffers, 1, 1)]
        stacks = [[buf[0], buf[0]] for buf in buffers]

        if hasattr(self, 'tracker'):
            self.tracker.reset_state()
        else:
            assert transitions is not None

        if transitions is not None:
            num_transitions = transitions.size(0)
            # trans_loss, trans_acc = 0, 0
        else:
            num_transitions = len(buffers[0]) * 2 - 3

        for i in range(num_transitions):
            if transitions is not None:
                trans = transitions[i]
            if hasattr(self, 'tracker'):
                tracker_states, trans_hyp = self.tracker(buffers, stacks)
                if trans_hyp is not None:
                    trans = trans_hyp.max(1)[1]
                    # if transitions is not None:
                    #     trans_loss += F.cross_entropy(trans_hyp, trans)
                    #     trans_acc += (trans_preds.data == trans.data).mean()
                    # else:
                    #     trans = trans_preds
            else:
                tracker_states = itertools.repeat(None)
            lefts, rights, trackings = [], [], []
            batch = zip(trans.data, buffers, stacks, tracker_states)
            for transition, buf, stack, tracking in batch:
                if transition == 3:  # shift
                    stack.append(buf.pop())
                elif transition == 2:  # reduce
                    rights.append(stack.pop())
                    lefts.append(stack.pop())
                    trackings.append(tracking)
            if rights:
                reduced = iter(self.reduce(lefts, rights, trackings))
                for transition, stack in zip(trans.data, stacks):
                    if transition == 2:
                        stack.append(next(reduced))
        # if trans_loss is not 0:
        return bundle([stack.pop() for stack in stacks])[0]

def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext SNLI example')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_embed', type=int, default=300)
    parser.add_argument('--d_proj', type=int, default=300)
    parser.add_argument('--d_hidden', type=int, default=300)
    parser.add_argument('--d_mlp', type=int, default=600)
    parser.add_argument('--n_mlp_layers', type=int, default=3)
    parser.add_argument('--d_tracker', type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--lr_decay_by', type=float, default=1)
    parser.add_argument('--lr_decay_every', type=float, default=1)
    parser.add_argument('--dev_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--embed_dropout', type=float, default=0.2)
    parser.add_argument('--mlp_dropout', type=float, default=0.2)
    parser.add_argument('--rnn_dropout', type=float, default=0.2)
    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--no-projection', action='store_false', dest='projection')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--predict_transitions', action='store_true', dest='predict')
    parser.add_argument('--spinn', action='store_true', dest='spinn')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--data_cache', type=str, default=os.path.join(os.getcwd(), '.data_cache'))
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
    parser.add_argument('--word_vectors', type=str, default='glove.42B')
    parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args


def train():
    # obtain command line parameters
    args = get_args()
    print(args)

    # set GPU node number

    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)

    if args.spinn:
        inputs = datasets.snli.ParsedTextField(lower=args.lower)
        transitions = datasets.snli.ShiftReduceField()
    else:
        inputs = data.Field(lower=args.lower)
        transitions = None

    answers = data.Field(sequential=False)

    # loads data from corpus
    train, dev, test = datasets.SNLI.splits(inputs, answers, transitions)

    inputs.build_vocab(train, dev, test)
    if args.word_vectors:
        if os.path.isfile(args.vector_cache):
            inputs.vocab.vectors = torch.load(args.vector_cache)
        else:
            inputs.vocab.load_vectors(vectors=args.word_vectors)
            os.makedirs(os.path.dirname(args.vector_cache), exist_ok=True)
            torch.save(inputs.vocab.vectors, args.vector_cache)
    answers.build_vocab(train)

    # Prepare iterator split
    train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test), batch_size=args.batch_size, device=args.gpu)

    config = args
    config.n_embed = len(inputs.vocab)
    config.d_out = len(answers.vocab)
    config.n_cells = config.n_layers

    # set hyperparameters

    if config.birnn:
        config.n_cells *= 2

    if config.spinn:
        config.lr = 2e-3 # 3e-4
        config.lr_decay_by = 0.75
        config.lr_decay_every = 1 #0.6
        config.regularization = 0 #3e-6
        config.mlp_dropout = 0.07
        config.embed_dropout = 0.08 # 0.17
        config.n_mlp_layers = 2
        config.d_tracker = 64
        config.d_mlp = 1024
        config.d_hidden = 300
        config.d_embed = 300
        config.d_proj = 600
        torch.backends.cudnn.enabled = False
    else:
        config.regularization = 0

    model = SNLIClassifier(config)
    if config.spinn:
        model.out[len(model.out._modules) - 1].weight.data.uniform_(-0.005, 0.005)
    if args.word_vectors:
        model.embed.weight.data = inputs.vocab.vectors
    if args.gpu != -1:
        model.cuda()
    if args.resume_snapshot:
        model.load_state_dict(torch.load(args.resume_snapshot))

    criterion = nn.CrossEntropyLoss()
    #opt = optim.Adam(model.parameters(), lr=args.lr)
    opt = optim.RMSprop(model.parameters(), lr=config.lr, alpha=0.9, eps=1e-6, weight_decay=config.regularization)

    iterations = 0
    start = time.time()
    best_dev_acc = -1
    train_iter.repeat = False
    header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
    dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
    log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
    os.makedirs(args.save_path, exist_ok=True)
    print(header)

    for epoch in range(args.epochs):
        train_iter.init_epoch()
        n_correct = n_total = train_loss = 0
        for batch_idx, batch in enumerate(train_iter):
            model.train(); opt.zero_grad()
            for pg in opt.param_groups:
                pg['lr'] = args.lr * (args.lr_decay_by ** (
                    iterations / len(train_iter) / args.lr_decay_every))
            iterations += 1 
            answer = model(batch)
            #print(nn.functional.softmax(answer[0]).data.tolist(), batch.label.data[0])
            n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
            n_total += batch.batch_size
            train_acc = 100. * n_correct/n_total
            loss = criterion(answer, batch.label)
            loss.backward(); opt.step(); train_loss += loss.data[0] * batch.batch_size
            if iterations % args.save_every == 0:
                snapshot_prefix = os.path.join(args.save_path, 'snapshot')
                snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, train_loss / n_total, iterations)
                torch.save(model.state_dict(), snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)
            if iterations % args.dev_every == 0:
                model.eval(); dev_iter.init_epoch()
                n_dev_correct = dev_loss = 0
                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                     answer = model(dev_batch)
                     n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                     dev_loss += criterion(answer, dev_batch.label).data[0] * dev_batch.batch_size
                dev_acc = 100. * n_dev_correct / len(dev)
                print(dev_log_template.format(time.time()-start,
                    epoch, iterations, 1+batch_idx, len(train_iter),
                    100. * (1+batch_idx) / len(train_iter), train_loss / n_total, dev_loss / len(dev), train_acc, dev_acc))
                n_correct = n_total = train_loss = 0
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
                    snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(dev_acc, dev_loss / len(dev), iterations)
                    torch.save(model.state_dict(), snapshot_path)
                    for f in glob.glob(snapshot_prefix + '*'):
                        if f != snapshot_path:
                            os.remove(f)
            elif iterations % args.log_every == 0:
                print(log_template.format(time.time()-start,
                    epoch, iterations, 1+batch_idx, len(train_iter),
                    100. * (1+batch_idx) / len(train_iter), train_loss / n_total, ' '*8, n_correct / n_total*100, ' '*12))
                n_correct = n_total = train_loss = 0

if __name__=="__main__":
    train()
