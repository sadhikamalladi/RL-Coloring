import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

USE_CUDA = True

class Attention(nn.Module):
    def __init__(self, hidden_size, use_tanh=False, C=10, name='Bahdanau', use_cuda=USE_CUDA):
        super(Attention, self).__init__()
        
        self.use_tanh = use_tanh
        self.C = C
        self.name = name
        
        if name == 'Bahdanau':
            self.W_query = nn.Linear(hidden_size, hidden_size)
            self.W_ref   = nn.Conv1d(hidden_size, hidden_size, 1, 1)

            V = torch.FloatTensor(hidden_size)
            if use_cuda:
                V = V.cuda()  
            self.V = nn.Parameter(V)
            self.V.data.uniform_(-(1. / math.sqrt(hidden_size)) , 1. / math.sqrt(hidden_size))
            
        
    def forward(self, query, ref):
        """
        Args: 
            query: [batch_size x hidden_size]
            ref:   ]batch_size x seq_len x hidden_size]
        """
        
        batch_size = ref.size(0)
        seq_len    = ref.size(1)
        
        if self.name == 'Bahdanau':
            ref = ref.permute(0, 2, 1)
            query = self.W_query(query).unsqueeze(2)  # [batch_size x hidden_size x 1]
            ref   = self.W_ref(ref)  # [batch_size x hidden_size x seq_len] 
            expanded_query = query.repeat(1, 1, seq_len) # [batch_size x hidden_size x seq_len]
            V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size x 1 x hidden_size]
            logits = torch.bmm(V, F.tanh(expanded_query + ref)).squeeze(1)
            
        elif self.name == 'Dot':
            query  = query.unsqueeze(2)
            logits = torch.bmm(ref, query).squeeze(2) #[batch_size x seq_len x 1]
            ref = ref.permute(0, 2, 1)
        
        else:
            raise NotImplementedError
        
        if self.use_tanh:
            logits = self.C * F.tanh(logits)
        else:
            logits = logits  
        return ref, logits

class GraphEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size, use_cuda=USE_CUDA):
        super(GraphEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.use_cuda = use_cuda
        
        self.embedding = nn.Parameter(torch.FloatTensor(input_size, embedding_size)) 
        self.embedding.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))
        
    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len    = inputs.size(2)
        embedding = self.embedding.repeat(batch_size, 1, 1)  
        embedded = []
        inputs = inputs.unsqueeze(1)
        for i in range(seq_len):
            embedded.append(torch.bmm(inputs[:, :, :, i].float(), embedding))
        embedded = torch.cat(embedded, 1)
        return embedded

class PointerNet(nn.Module):
    def __init__(self, hps, num_nodes, use_cuda=USE_CUDA):
        super(PointerNet, self).__init__()
        
        self.embedding_size = hps.embedding_size
        self.hidden_size    = hps.hidden_size
        self.n_glimpses     = hps.n_glimpses
        self.use_cuda       = use_cuda
        self.attention      = hps.attention_type
        
        
        self.embedding = GraphEmbedding(num_nodes, self.embedding_size, use_cuda=use_cuda)
        self.encoder = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True)
        self.decoder = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True)
        self.pointer = Attention(self.hidden_size, use_tanh=hps.use_tanh, C=hps.tanh_exploration, name=self.attention, use_cuda=use_cuda)
        self.glimpse = Attention(self.hidden_size, use_tanh=False, name=self.attention, use_cuda=use_cuda)
        
        self.decoder_start_input = nn.Parameter(torch.FloatTensor(self.embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(self.embedding_size)), 1. / math.sqrt(self.embedding_size))
        
    def apply_mask_to_logits(self, logits, mask, idxs): 
        batch_size = logits.size(0)
        clone_mask = mask.clone()

        if idxs is not None:
            clone_mask[[i for i in range(batch_size)], idxs.data] = 1
            logits[clone_mask] = -np.inf
        return logits, clone_mask
            
    def forward(self, inputs):
        """
        Args: 
            inputs: [batch_size x 1 x sourceL]
        """
        batch_size = inputs.size(0)
        seq_len    = inputs.size(2)
        
        embedded = self.embedding(inputs)
        encoder_outputs, (hidden, context) = self.encoder(embedded)
        
        
        prev_probs = []
        prev_idxs = []
        mask = torch.zeros(batch_size, seq_len).byte()
        if self.use_cuda:
            mask = mask.cuda()
            
        idxs = None
       
        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)
        
        for i in range(seq_len):
            
            
            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))
            
            query = hidden.squeeze(0)
            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
                query = torch.bmm(ref, F.softmax(logits).unsqueeze(2)).squeeze(2) 
                
                
            _, logits = self.pointer(query, encoder_outputs)
            logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
            probs = F.softmax(logits)
            
            idxs = torch.multinomial(probs, 1).squeeze(1)
            for old_idxs in prev_idxs:
                if old_idxs.eq(idxs).data.any():
                    idxs = torch.multinomial(probs, 1).squeeze(1)
                    break

            decoder_input = embedded[[i for i in range(batch_size)], idxs.data, :]
            
            prev_probs.append(probs)
            prev_idxs.append(idxs)
            
        return prev_probs, prev_idxs

class CombinatorialRL(nn.Module):
    def __init__(self, hps, use_cuda=USE_CUDA):
        super(CombinatorialRL, self).__init__()
        self.use_cuda = use_cuda
        
        self.actor = PointerNet(hps, use_cuda)

    def reward(self, ordering, graph, USE_CUDA=False):
        batch_size = ordering.shape[0]
        n = ordering.shape[1]
        coloring_reward = Variable(torch.zeros([batch_size]))
        num_colors = torch.zeros([batch_size])
    
        if USE_CUDA:
            coloring_reward = coloring_reward.cuda()

        for b in range(batch_size):
            coloring = np.zeros(n) - 1
            reward = 0
            for i in range(n):
                v = ordering[b, i].item()
                available_colors = np.ones(n)
                neighbors = graph[b].neighbors(v)
                for neighbor in neighbors:
                    if coloring[neighbor] != -1:
                        available_colors[int(coloring[neighbor])] = 0
                min_color = np.where(available_colors==1)[0][0]
                max_color = np.max(coloring)
                # we do the inverse of what we expect as rewards for adding
                # a new color because model trains to minimize rewards
                if min_color > max_color:
                    reward += 1
                else:
                    reward -= 1
                coloring[v] = min_color
            coloring_reward[b] = reward
            num_colors[b] = np.max(coloring) + 1
        
        return coloring_reward, num_colors

    def forward(self, inputs, original_graph):
        """
        Args:
            inputs: [batch_size, input_size, seq_len]
        """
        batch_size = inputs.size(0)
        input_size = inputs.size(1)
        seq_len    = inputs.size(2)
        
        probs, action_idxs = self.actor(inputs)
            
        action_probs = []    
        for prob, action_id in zip(probs, action_idxs):
            action_probs.append(prob[[x for x in range(batch_size)], action_id.data])

        acts = torch.stack(action_idxs).transpose(1,0)

        R, colors = self.reward(acts, original_graph, self.use_cuda)
        
        return R,  action_idxs, action_probs, colors
