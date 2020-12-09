import torch
from torch import nn
from torch.autograd import Variable


# LSTM
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):

        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self._W = nn.Parameter(torch.FloatTensor(input_size + hidden_size, 4 * hidden_size))
        self._b = nn.Parameter(torch.FloatTensor(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._W.data)
        nn.init.constant_(self._b.data, 0)

    def forward(self, x, s_):
        h_, c_ = s_

        candidate = torch.mm(torch.cat([x, h_], -1), self._W) + self._b
        i, f, o, g = candidate.split(self.hidden_size, -1)

        c = i.sigmoid() * g.tanh() + f.sigmoid() * c_
        h = o.sigmoid() * c.tanh()

        return h, c


# GRU
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):

        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self._W = nn.Parameter(torch.FloatTensor(input_size + hidden_size, 2 * hidden_size))
        self._W_b = nn.Parameter(torch.FloatTensor(2 * hidden_size))
        self._U = nn.Parameter(torch.FloatTensor(input_size + hidden_size, hidden_size))
        self._U_b = nn.Parameter(torch.FloatTensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._W.data)
        nn.init.xavier_uniform_(self._U.data)
        nn.init.constant_(self._W_b.data, 0)
        nn.init.constant_(self._U_b.data, 0)

    def forward(self, x, h_):

        g = torch.mm(torch.cat([x, h_], -1), self._W) + self._W_b

        r, u = g.sigmoid().split(self.hidden_size, -1)

        c = torch.mm(torch.cat([x, r * h_], -1), self._U) + self._U_b

        h = u * h_ + (1. - u) * c.tanh()

        return h


# ATR
class ATRCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ATRCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self._W = nn.Parameter(torch.FloatTensor(input_size, hidden_size))
        self._W_b = nn.Parameter(torch.FloatTensor(hidden_size))
        self._U = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size))
        self._U_b = nn.Parameter(torch.FloatTensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._W.data)
        nn.init.xavier_uniform_(self._U.data)
        nn.init.constant_(self._W_b.data, 0)
        nn.init.constant_(self._U_b.data, 0)

    def forward(self, x, h_):

        p = torch.mm(x, self._W) + self._W_b
        q = torch.mm(h_, self._U) + self._U_b

        i = (p + q).sigmoid()
        f = (p - q).sigmoid()

        h = (i * p + f * h_).tanh()

        return h


# LRN
class LRNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LRNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self._W = nn.Parameter(torch.FloatTensor(input_size, hidden_size * 3))
        self._W_b = nn.Parameter(torch.FloatTensor(hidden_size * 3))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._W.data)
        nn.init.constant_(self._W_b.data, 0)

    def forward(self, x, h_):

        p, q, r = (torch.mm(x, self._W) + self._W_b).split(self.hidden_size, -1)

        i = (p + h_).sigmoid()
        f = (q - h_).sigmoid()

        h = (i * r + f * h_).tanh()

        return h


# SRU
class SRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):

        super(SRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self._W = nn.Parameter(torch.FloatTensor(input_size, 4 * hidden_size))
        self._W_b = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        self._Vr = nn.Parameter(torch.FloatTensor(hidden_size))
        self._Vf = nn.Parameter(torch.FloatTensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._W.data)
        nn.init.constant_(self._Vr.data, 1)
        nn.init.constant_(self._Vf.data, 1)
        nn.init.constant_(self._W_b.data, 0)

    def forward(self, x, s_):
        h_, c_ = s_

        g = torch.mm(x, self._W) + self._W_b

        g1, g2, g3, g4 = g.split(self.hidden_size, -1)

        f = (g1 + self._Vf * c_).sigmoid()
        c = f * c_ + (1. - f) * g2
        r = (g3 + self._Vr * c_).sigmoid()
        h = r * c + (1. - r) * g4

        return h, c


def get_cell(cell_type):
    cell_type = cell_type.lower()

    print("RNN Type: **{}**".format(cell_type))

    if cell_type == "gru":
        cell = GRUCell
    elif cell_type == "lstm":
        cell = LSTMCell
    elif cell_type == "atr":
        cell = ATRCell
    elif cell_type == "lrn":
        cell = LRNCell
    elif cell_type == "sru":
        cell = SRUCell
    else:
        raise NotImplementedError(
            "{} is not supported".format(cell_type))

    return cell


class RNN(nn.Module):

    def __init__(self, cell_type, input_size, hidden_size,
                 num_layers=1, batch_first=False, dropout=0, n_students=0, **kwargs):
        super(RNN, self).__init__()
        self.cell_type = cell_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.n_students = n_students
        self.c_on = self.cell_type == "lstm" or self.cell_type == "sru"

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell_type = get_cell(cell_type)
            if n_students > 0:
                cell = StudentDistillCell(input_size=layer_input_size, hidden_size=hidden_size, cell_type=cell_type,
                                          n_students=n_students, c_on=self.c_on, **kwargs) 
            else:
                cell = cell_type(input_size=layer_input_size, hidden_size=hidden_size, **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    def _forward_rnn(self, cell, x, h, L, state_post_proc=None, **kwargs):
        max_time = x.size(0)
        output = []
        if self.n_students == 0:
            cell_kwargs = dict(kwargs)
            del cell_kwargs['distill_loss_acc']
        else:
            cell_kwargs = kwargs
        for time in range(max_time):
            if self.c_on:
                new_h, new_c = cell(x[time], h, **cell_kwargs)
            else:
                new_h = cell(x[time], h, **cell_kwargs)

            mask = (time < L).float().unsqueeze(1).expand_as(new_h)
            new_h = new_h*mask + h[0]*(1 - mask)

            if self.c_on:
                new_c = new_c*mask + h[1]*(1 - mask)
                h = (new_h, new_c)
            else:
                h = new_h

            if state_post_proc is not None:
                new_h = state_post_proc(new_h, **kwargs)
            output.append(new_h)

        output = torch.stack(output, 0)
        return output, h

    def forward(self, x, h=None, L=None, state_post_proc=None, **kwargs):
        if self.batch_first:
            x = x.transpose(0, 1)
        max_time, batch_size, _ = x.size()
        if L is None:
            L = Variable(torch.LongTensor([max_time] * batch_size))
            if x.is_cuda:
                L = L.cuda(x.get_device())
        if h is None:
            if self.c_on:
                h = (Variable(nn.init.xavier_uniform(x.new(self.num_layers, batch_size, self.hidden_size))),
                     Variable(nn.init.xavier_uniform(x.new(self.num_layers, batch_size, self.hidden_size))))
            else:
                h = Variable(nn.init.xavier_uniform(x.new(self.num_layers, batch_size, self.hidden_size)))

        layer_output = None
        states = []
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            if self.c_on:
                h_layer = (h[0][layer, :, :], h[1][layer, :, :])
            else:
                h_layer = h[layer, :, :]
            
            #state_post_proc_cur_layer = state_post_proc[layer] if state_post_proc is not None else None
            state_post_proc_cur_layer = state_post_proc
            if layer == 0:
                layer_output, layer_state = self._forward_rnn(
                    cell, x, h_layer, L, state_post_proc=state_post_proc_cur_layer, **kwargs)
            else:
                layer_output, layer_state = self._forward_rnn(
                    cell, layer_output, h_layer, L, state_post_proc=state_post_proc_cur_layer, **kwargs)

            if layer != self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)
            states.append(layer_state)

        output = layer_output

        if self.c_on:
            states = list(zip(*states))
            return output, (torch.stack(states[0], 0), torch.stack(states[1], 0))
        else:
            return output, torch.stack(states, 0)

# LM robustness
class StudentDistillCell(nn.Module):
    def __init__(self, input_size, hidden_size, cell_type, n_students=0, c_on=False):

        super(StudentDistillCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.n_students = n_students
        self.c_on = c_on
        
        self.main_cell = cell_type(input_size, hidden_size)
        self.student_cells = torch.nn.ModuleList([cell_type(input_size, hidden_size) for i in range(n_students)])
        self.distill_loss = torch.nn.MSELoss(reduction='mean')

    def reset_parameters(self):
        self.main_cell.reset_parameters()
        for i in range(self.n_students):
            self.student_cells[i].reset_parameters()

    def forward(self, x, h_, average_ensemble=False, distill_loss_acc=None):
        h = self.main_cell(x, h_)
        x_nograd, h__nograd = x.detach(), h_.detach()
        student_h_list = [student_cell(x_nograd, h__nograd) for student_cell in self.student_cells]
        if distill_loss_acc != None:
            for student_h in student_h_list:
                if self.c_on:
                    student_distill_loss = self.distill_loss(student_h[0], h[0].detach())
                    distill_loss_acc[0] += student_distill_loss
                    student_distill_loss = self.distill_loss(student_h[1], h[1].detach())
                    distill_loss_acc[0] += student_distill_loss
                else:
                    student_distill_loss = self.distill_loss(student_h, h.detach())
                    distill_loss_acc[0] += student_distill_loss
        if average_ensemble:
            all_h_list = student_h_list + [h]
            if self.c_on:
                all_hh_list = [h[0] for h in all_h_list]
                all_hc_list = [h[1] for h in all_h_list]
                hh = torch.stack(all_hh_list).mean(dim=0)
                hc = torch.stack(all_hc_list).mean(dim=0)
                h = (hh, hc)
            else:
                h = torch.stack(all_h_list).mean(dim=0)
        return h

#class StatePostProcCell(nn.Module):
#    def __init__(self, main_cell, postproc_module=None):
#        super(StatePostProcCell, self).__init__()
#        self.main_cell=main_cell
#        self.postproc_module=postproc_module
#        
#    def reset_parameters(self):
#        self.main_cell.reset_parameters()
#    
#    def forward(self, x, h_, postproc_args={}, **kwargs):
#        h = self.main_cell(x, h_)
#        if self.postproc_module is not None:
#            h = self.postproc_module(h, **postproc_args)
#        return h
