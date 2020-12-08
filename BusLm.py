import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import math
from random import random

class AdditiveAttention(nn.Module):
    ''' AttentionPooling used to weighted aggregate news vectors
    Arg:
        d_h: the last dimension of input
    '''

    def __init__(self, d_h, hidden_size=200):
        super(AdditiveAttention, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)


    def forward(self, x, attn_mask=None):
        """
        Args:
            x: batch_size, candidate_size, candidate_vector_dim
            attn_mask: batch_size, candidate_size
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)

        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  # (bz, 400)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        #       [bz, 20, seq_len, 20] x [bz, 20, 20, seq_len] -> [bz, 20, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * (attn_mask.unsqueeze(1).unsqueeze(1))
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)

        #       [bz, 20, seq_len, seq_len] x [bz, 20, seq_len, 20] -> [bz, 20, seq_len, 20]
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, enable_gpu):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # 300
        self.n_heads = n_heads  # 20
        self.d_k = d_k  # 20
        self.d_v = d_v  # 20
        self.enable_gpu = enable_gpu

        self.W_Q = nn.Linear(d_model, d_k * n_heads)  # 300, 400
        self.W_K = nn.Linear(d_model, d_k * n_heads)  # 300, 400
        self.W_V = nn.Linear(d_model, d_v * n_heads)  # 300, 400

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K, V, mask=None):
        #       Q, K, V: [bz, seq_len, 300] -> W -> [bz, seq_len, 400]-> q_s: [bz, 20, seq_len, 20]
        _, batch_size = Q, Q.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads,
                               self.d_v).transpose(1, 2)

        context, attn = ScaledDotProductAttention(self.d_k)(
            q_s, k_s, v_s, mask)  # [bz, 20, seq_len, 20]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_v)  # [bz, seq_len, 400]
        #         output = self.fc(context)
        return context  # self.layer_norm(output + residual)



class TextEncoder(torch.nn.Module):
    def __init__(self,
                 bert_model,
                 bus_num,
                 seg_num,
                 word_embedding_dim,
                 num_attention_heads,
                 query_vector_dim,
                 dropout_rate,
                 bert_layer_hidden=None,
                 enable_gpu=True):
        '''
        seg_num: the number of segments
        bus_num: =seg_num if use buslm and seg_num>1 else 0
        '''
        super(TextEncoder, self).__init__()
        self.bert_model = bert_model
        self.bus_num = bus_num
        self.seg_num = seg_num
        self.dropout_rate = dropout_rate
        self.bert_layer_hidden = bert_layer_hidden
        self.multihead_attention = MultiHeadAttention(word_embedding_dim,
                                                      num_attention_heads, 20,
                                                      20, enable_gpu)
        self.additive_attention = AdditiveAttention(num_attention_heads * 20,
                                                    query_vector_dim)


    def forward(self,segments,token_masks,seg_masks):
        """
        Args:
            segments: seg_num * Tensor(batch_size, num_words) e.g., (seg1_ids,seg2_ids,seg3_ids)
            token_masks: (seg1_mask, seg2_mask, seg3_mask)
            seg_masks:  mask the segments which are empty
        Returns:
            (shape) seg_num * (batch_size, word_embedding_dim)
        """

        if self.bus_num>0:
        # For each segment, concatenate it and all [CLS](including its [CLS]), so need to mask one of its [CLS]
            attention_mask = ()
            for seg_idx, mask in enumerate(token_masks):
                mask = torch.cat([seg_masks, mask], -1)
                mask[:,seg_idx] = 0
                attention_mask = attention_mask + (mask,)
            token_masks = attention_mask

        last_hidden_states = self.bert_model(segments, token_masks)[0] #num_segments * (batch_size, num_words, hidden dim)

        text_vectors = ()
        if self.bus_num>0:
            #concatenate all [CLS]
            station_emb = [last_hidden_states[idx][:, :1] for idx in range(self.bus_num)]
            station_emb = torch.cat(station_emb, dim=1)

            for seg_idx in range(self.seg_num):
                seg_hidden_state = torch.cat([station_emb, last_hidden_states[seg_idx]], -2)
                seg_embed = F.dropout(seg_hidden_state,p=self.dropout_rate,training=self.training)
                propogation = self.multihead_attention(
                    seg_embed,seg_embed,seg_embed, token_masks[seg_idx])

                text_vec = F.dropout(propogation, p=self.dropout_rate, training=self.training)
                text_vec = self.additive_attention(text_vec[:, self.bus_num:], token_masks[seg_idx][:, self.bus_num:])
                text_vectors = text_vectors + (text_vec,)

        else:
            for seg_idx in range(self.seg_num):
                seg_embed = F.dropout(last_hidden_states[seg_idx], p=self.dropout_rate, training=self.training)

                propogation = self.multihead_attention(
                    seg_embed, seg_embed, seg_embed, token_masks[seg_idx])

                text_vec = F.dropout(propogation, p=self.dropout_rate, training=self.training)
                text_vec = self.additive_attention(text_vec, token_masks[seg_idx])
                text_vectors = text_vectors + (text_vec,)

        return text_vectors


class ElementEncoder(torch.nn.Module):
    def __init__(self, num_elements, embedding_dim, enable_gpu=True):
        super(ElementEncoder, self).__init__()
        self.enable_gpu = enable_gpu
        self.embedding = nn.Embedding(num_elements,
                                      embedding_dim,
                                      padding_idx=0)

    def forward(self, element):
        element_vector = self.embedding(element)
        return element_vector


class NewsEncoder(torch.nn.Module):
    def __init__(self, args, bert_model, category_dict_size,
                 domain_dict_size, subcategory_dict_size):
        super(NewsEncoder, self).__init__()
        self.args = args

        self.text_encoders = TextEncoder(bert_model,
                            args.bus_num,
                            args.seg_num,
                            args.word_embedding_dim,
                            args.num_attention_heads, args.news_query_vector_dim,
                            args.drop_rate, args.bert_layer_hidden, args.enable_gpu)

        element_encoders_candidates = ['category', 'domain', 'subcategory']
        element_encoders = set(args.news_attributes) & set(element_encoders_candidates)

        name2num = {
            "category": category_dict_size + 1,
            "domain": domain_dict_size + 1,
            "subcategory": subcategory_dict_size + 1
        }
        self.element_encoders = nn.ModuleDict({
            name: ElementEncoder(name2num[name],
                                 args.num_attention_heads * 20,
                                 args.enable_gpu)
            for name in (element_encoders)
        })

        if len(args.news_attributes) > 1:
            self.final_attention = AdditiveAttention(
                args.num_attention_heads * 20, args.news_query_vector_dim)

        self.reduce_dim_linear = nn.Linear(args.num_attention_heads * 20,
                                           args.news_dim)


    def forward(self, segments, token_masks, seg_masks, elements=None):
        """
        Args:
            segments: seg_num * Tensor(batch_size, num_words)
        Returns:
            (shape) batch_size, news_dim
        """
        all_vectors = self.text_encoders(segments,token_masks,seg_masks)

        if 'body' in self.args.news_attributes:
            body_vec = torch.stack(all_vectors[-self.args.body_seg_num:],dim=1)
            body_vec = torch.mean(body_vec,dim=1)
            all_vectors = all_vectors[:self.args.seg_num-self.args.body_seg_num] + (body_vec,)


        for idx, name in enumerate(self.element_encoders):
            ele_vec = self.element_encoders[name](elements[:,idx])
            all_vectors = all_vectors + (ele_vec,)


        if len(all_vectors) == 1:
            final_news_vector = all_vectors[0]
        else:
            final_news_vector = self.final_attention(
                torch.stack(all_vectors, dim=1))

        final_news_vector = self.reduce_dim_linear(final_news_vector)
        return final_news_vector


class UserEncoder(torch.nn.Module):
    def __init__(self, args):
        super(UserEncoder, self).__init__()
        self.args = args
        self.news_additive_attention = AdditiveAttention(
            args.news_dim, args.user_query_vector_dim)

        self.att_fc1 = nn.Linear(self.args.news_dim, args.user_query_vector_dim)
        self.att_fc2 = nn.Linear(args.user_query_vector_dim, 1)


    def _process_news(self, vec, mask, pad_doc,
                      min_log_length=1):

        if self.args.add_pad:
            padding_doc = pad_doc.expand(vec.shape[0], self.args.news_dim).unsqueeze(1)
            vec = torch.cat([padding_doc,vec],1)
            min_log_length += 1

        # batch_size:B, log_length:L+1, news_dim:D, predicte item num: G
        vec = vec[:,:-1,:]
        B,L,D = vec.shape      #B L D
        min_log_length = min(min_log_length,L)
        G = L+1 - min_log_length

        autor_mask = torch.ones((L, G), dtype=torch.float).triu(1 - min_log_length).transpose(0, 1).to(vec.device) # G L
        auto_mask = autor_mask.unsqueeze(0).repeat(B, 1, 1)  #B G L

        if not self.args.use_pad:
            auto_mask = torch.mul(autor_mask,mask[:,:-1].unsqueeze(2).repeat(1,1,L)) #B G L

        vec_repeat = vec.unsqueeze(1).repeat(1, G, 1, 1) #B G L L

        weights = self.att_fc1(vec_repeat)
        weights = nn.Tanh()(weights)
        weights = self.att_fc2(weights).squeeze(3)
        weights = weights.masked_fill(auto_mask == 0, -1e8)
        weights = torch.softmax(weights, dim=-1) #B G L
        user_vec = torch.matmul(weights, vec)  #B G D

        return user_vec


    def infer_user_vec(self,log_vec, log_mask):
        weights = self.att_fc1(log_vec)
        weights = nn.Tanh()(weights)
        weights = self.att_fc2(weights).squeeze(2)
        weights = weights.masked_fill(log_mask == 0, -1e8)
        weights = torch.softmax(weights, dim=-1)
        user_vec = torch.matmul(weights.unsqueeze(1),log_vec).squeeze(1)
        return user_vec


    def forward(self, log_vec, log_mask, pad_embedding):
        """
        Returns:
            (shape) batch_size, predict_num, news_dim
        """
        log_vec = self._process_news(log_vec, log_mask, pad_embedding)

        return log_vec



class ModelBert(torch.nn.Module):
    def __init__(self,
                 args,
                 bert_model,
                 category_dict_size=0,
                 domain_dict_size=0,
                 subcategory_dict_size=0):
        super(ModelBert, self).__init__()
        self.args = args
        #
        self.news_encoder = NewsEncoder(args,
                                        bert_model,
                                        category_dict_size,
                                        domain_dict_size,
                                        subcategory_dict_size)
        self.user_encoder = UserEncoder(args)
        self.pad_doc = nn.Parameter(torch.empty(1, args.news_dim).uniform_(-1, 1)).type(torch.FloatTensor)  #nn.Parameter default requires_grad=True
        #
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,
                segments,
                token_masks,
                seg_masks,
                elements,
                cache_vec,
                batch_hist,
                batch_mask,
                batch_negs
                ):
        """
        args:
            segments:    seg_num * Tensor(batch_size, num_words)
            token_masks: mask padding token
            seg_masks:   mask the empty segment
            elements:    [category, subcategory, doamin]
            cache_vec:  the index of embedding from cache
            batch_hist:  user history
            batch_mask:  mask of user history
            batch_negs:  user num * history lenght-1 * npratio
        Returns:
            loss,new_encoded_embedding
        """
        encode_vecs = self.news_encoder(segments, token_masks, seg_masks, elements)
        if cache_vec is not None:
            news_vecs = torch.cat([self.pad_doc,cache_vec,encode_vecs],0)
        else:
            news_vecs = torch.cat([self.pad_doc, encode_vecs], 0)

        reshape_negs = batch_negs.view(-1,)
        neg_vec = news_vecs.index_select(0,reshape_negs)
        neg_vec = neg_vec.view(batch_negs.size(0),batch_negs.size(1),batch_negs.size(2),-1) #B G N D

        reshape_hist = batch_hist.view(-1,)
        log_vec = news_vecs.index_select(0,reshape_hist)
        log_vec = log_vec.view(batch_hist.size(0),batch_hist.size(1),-1) # batch_size, log_length, news_dim

        user_vector = self.user_encoder(log_vec, batch_mask, self.pad_doc) #B G D

        candidate = torch.cat([log_vec[:,1:,:].unsqueeze(2),neg_vec],2) #B G 1+N D

        score = torch.matmul(candidate, user_vector.unsqueeze(-1)).squeeze(dim=-1) #B G 1+N
        logits = F.softmax(score, -1)
        loss = -torch.log(logits[:, :, 0] + 1e-9)
        loss = torch.mul(loss, batch_mask[:, 1:])
        loss = torch.sum(loss)/(torch.sum(batch_mask[:, 1:])+1e-9)
        return loss,encode_vecs



if __name__ == "__main__":
    from parameters import parse_args
    from tnlrv3.bus_modeling  import BusLM_rec
    import utils
    from tnlrv3.modeling import TuringNLRv3ForSequenceClassification
    from tnlrv3.configuration_tnlrv3 import TuringNLRv3Config
    from tnlrv3.tokenization_tnlrv3 import TuringNLRv3Tokenizer

    utils.setuplogging()


    args = parse_args()
    args.news_attributes = ['title', 'abstract', 'body','catagory','subcategory','domain']  # , 'abstract', 'category', 'domain', 'subcategory']
    args.news_dim = 5
    args.bus_connection = True
    args.body_seg_num = 2

    seg_num = 0
    for name in args.news_attributes:
        if name == 'title':
            seg_num += 1
        elif name == 'abstract':
            seg_num += 1
        elif name == 'body':
            seg_num += args.body_seg_num

    if seg_num>1 and args.bus_connection:
        bus_num = seg_num
    else:
        bus_num = 0

    args.config_name = "../Turing/buslm-base-uncased-config.json"
    MODEL_CLASSES = {
        'tnlrv3': (TuringNLRv3Config, BusLM_rec, TuringNLRv3Tokenizer),
    }
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        output_hidden_states=True)
    print(type(config))

    args.bus_num = bus_num
    config.bus_num = bus_num
    bert_model = model_class.from_pretrained(args.model_name_or_path,
                                             from_tf=bool('.ckpt' in args.model_name_or_path),
                                             config=config)


    # print(bert_model.bert.rel_pos_bias.weight)

    model = ModelBert(args, bert_model, 10, 10, 10)
    model.cuda()

    title_l, abs_l,body_l = 32,64,5

    title = torch.randint(0,3,(5, title_l)).cuda().long()
    abs = torch.randint(0,3,(5, abs_l)).cuda().long()
    body1 = torch.randint(0,3,(5, body_l)).cuda().long()
    body2 = torch.randint(0,3,(5, body_l)).cuda().long()

    title_mask = torch.randint(0,2,(5, title_l)).cuda().float()
    abs_mask = torch.randint(0,2,(5, abs_l)).cuda().float()
    body1_mask = torch.randint(0,2,(5, body_l)).cuda().float()
    body2_mask = torch.randint(0,2,(5, body_l)).cuda().float()

    seg_masks = torch.randint(0,2,(5, 4)).cuda().float()

    elements = torch.randint(0,3,(5, 3)).cuda().long()

    # batch_news_feature = ((title,abs,body1,body2),(title_mask,abs_mask,body1_mask,body2_mask),seg_masks,elements)
    #
    # batch_hist = torch.randint(0,10,(2, 3)).cuda().long()  # user num  history lenght
    # batch_mask = torch.ones((2, 3)).cuda().float() # user num  history lenght
    # batch_negs = torch.randint(0,10,(2, 2, 1)).cuda().long()
    #
    # get_cache = torch.randint(0,10000,(4,)).cuda().long()
    # # the index of embedding from cache
    # update_cache = [1]*5 + [0]*(10000-5)
    # update_cache = torch.tensor(update_cache).bool()


    print(model((title,abs,body1,body2),(title_mask,abs_mask,body1_mask,body2_mask),seg_masks,elements,None,None,None,None))

