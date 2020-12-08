from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

from transformers.modeling_bert import \
    BertPreTrainedModel, BertSelfOutput, BertIntermediate, \
    BertOutput, BertPredictionHeadTransform, BertPooler
from transformers.file_utils import WEIGHTS_NAME

from tnlrv3.config import TuringNLRv3ForSeq2SeqConfig
from tnlrv3.convert_state_dict import get_checkpoint_from_transformer_cache, state_dict_convert

logger = logging.getLogger(__name__)

BertLayerNorm = torch.nn.LayerNorm

TuringNLRv3_PRETRAINED_MODEL_ARCHIVE_MAP = {
}


class TuringNLRv3PreTrainedModel(BertPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = TuringNLRv3ForSeq2SeqConfig
    supported_convert_pretrained_model_archive_map = {
        "tnlrv3": TuringNLRv3_PRETRAINED_MODEL_ARCHIVE_MAP,
    }
    base_model_prefix = "TuringNLRv3_for_seq2seq"
    pretrained_model_archive_map = {
        **TuringNLRv3_PRETRAINED_MODEL_ARCHIVE_MAP,
    }

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(
            cls, pretrained_model_name_or_path, reuse_position_embedding=None,
            replace_prefix=None, *model_args, **kwargs,
    ):
        model_type = kwargs.pop('model_type', 'tnlrv3')
        if model_type is not None and "state_dict" not in kwargs:
            if model_type in cls.supported_convert_pretrained_model_archive_map:
                pretrained_model_archive_map = cls.supported_convert_pretrained_model_archive_map[model_type]
                if pretrained_model_name_or_path in pretrained_model_archive_map:
                    state_dict = get_checkpoint_from_transformer_cache(
                        archive_file=pretrained_model_archive_map[pretrained_model_name_or_path],
                        pretrained_model_name_or_path=pretrained_model_name_or_path,
                        pretrained_model_archive_map=pretrained_model_archive_map,
                        cache_dir=kwargs.get("cache_dir", None), force_download=kwargs.get("force_download", None),
                        proxies=kwargs.get("proxies", None), resume_download=kwargs.get("resume_download", None),
                    )
                    state_dict = state_dict_convert[model_type](state_dict)
                    kwargs["state_dict"] = state_dict
                    logger.info("Load HF ckpts")
                elif os.path.isfile(pretrained_model_name_or_path):
                    state_dict = torch.load(pretrained_model_name_or_path, map_location='cpu')
                    kwargs["state_dict"] = state_dict_convert[model_type](state_dict)
                    logger.info("Load local ckpts")
                elif os.path.isdir(pretrained_model_name_or_path):
                    state_dict = torch.load(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME),
                                            map_location='cpu')
                    kwargs["state_dict"] = state_dict_convert[model_type](state_dict)
                    logger.info("Load local ckpts")
                else:
                    raise RuntimeError("Not fined the pre-trained checkpoint !")

        if kwargs["state_dict"] is None:
            logger.info("TNLRv3 does't support the model !")
            raise NotImplementedError()

        config = kwargs["config"]
        state_dict = kwargs["state_dict"]
        # initialize new position embeddings (From Microsoft/UniLM)
        _k = 'bert.embeddings.position_embeddings.weight'
        if _k in state_dict:
            if config.max_position_embeddings > state_dict[_k].shape[0]:
                logger.info("Resize > position embeddings !")
                old_vocab_size = state_dict[_k].shape[0]
                new_postion_embedding = state_dict[_k].data.new_tensor(torch.ones(
                    size=(config.max_position_embeddings, state_dict[_k].shape[1])), dtype=torch.float)
                new_postion_embedding = nn.Parameter(data=new_postion_embedding, requires_grad=True)
                new_postion_embedding.data.normal_(mean=0.0, std=config.initializer_range)
                max_range = config.max_position_embeddings if reuse_position_embedding else old_vocab_size
                shift = 0
                while shift < max_range:
                    delta = min(old_vocab_size, max_range - shift)
                    new_postion_embedding.data[shift: shift + delta, :] = state_dict[_k][:delta, :]
                    logger.info("  CP [%d ~ %d] into [%d ~ %d]  " % (0, delta, shift, shift + delta))
                    shift += delta
                state_dict[_k] = new_postion_embedding.data
                del new_postion_embedding
            elif config.max_position_embeddings < state_dict[_k].shape[0]:
                logger.info("Resize < position embeddings !")
                old_vocab_size = state_dict[_k].shape[0]
                new_postion_embedding = state_dict[_k].data.new_tensor(torch.ones(
                    size=(config.max_position_embeddings, state_dict[_k].shape[1])), dtype=torch.float)
                new_postion_embedding = nn.Parameter(data=new_postion_embedding, requires_grad=True)
                new_postion_embedding.data.normal_(mean=0.0, std=config.initializer_range)
                new_postion_embedding.data.copy_(state_dict[_k][:config.max_position_embeddings, :])
                state_dict[_k] = new_postion_embedding.data
                del new_postion_embedding

        _k2 = 'bert.rel_pos_bias.weight'
        if _k2 in state_dict and state_dict[_k2].shape[1]!=(config.bus_num+config.rel_pos_bins):
            logger.info(f"rel_pos_bias.weight.shape[1]:{state_dict[_k2].shape[1]} != config.bus_num+config.rel_pos_bins:{config.bus_num+config.rel_pos_bins}")
            old_rel_pos_bias = state_dict[_k2]
            new_rel_pos_bias = torch.cat([old_rel_pos_bias,old_rel_pos_bias[:,-1:].expand(old_rel_pos_bias.size(0),config.bus_num)],-1)
            new_rel_pos_bias = nn.Parameter(data=new_rel_pos_bias, requires_grad=True)
            state_dict[_k2] = new_rel_pos_bias.data
            del new_rel_pos_bias

        #     if cofig

        if replace_prefix is not None:
            new_state_dict = {}
            for key in state_dict:
                if key.startswith(replace_prefix):
                    new_state_dict[key[len(replace_prefix):]] = state_dict[key]
                else:
                    new_state_dict[key] = state_dict[key]
            kwargs["state_dict"] = new_state_dict
            del state_dict

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        fix_word_embedding = getattr(config, "fix_word_embedding", None)
        if fix_word_embedding:
            self.word_embeddings.weight.requires_grad = False
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        else:
            self.token_type_embeddings = None

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        all_embeddings = []; all_posotion_ids = []
        device =  input_ids[0].device
        for seg_ids in input_ids:
            input_shape = seg_ids.shape

            position_ids = torch.arange(input_shape[1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

            inputs_embeds = self.word_embeddings(seg_ids)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = inputs_embeds + position_embeddings

            if self.token_type_embeddings:
                embeddings = embeddings + self.token_type_embeddings(token_type_ids)

            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            all_embeddings.append(embeddings)
            all_posotion_ids.append(position_ids)
        return all_embeddings, all_posotion_ids


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def multi_head_attention(self, query, key, value, attention_mask, rel_pos):
        query_layer = self.transpose_for_scores(query)
        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        if rel_pos is not None:
            attention_scores = attention_scores + rel_pos

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return (context_layer, attention_probs) if self.output_attentions else (context_layer,)

    def forward(self, hidden_states, attention_mask=None,
                encoder_hidden_states=None,
                split_lengths=None, rel_pos=None):
        mixed_query_layer = self.query(hidden_states)
        if split_lengths:
            assert not self.output_attentions

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        if split_lengths:
            query_parts = torch.split(mixed_query_layer, split_lengths, dim=1)
            key_parts = torch.split(mixed_key_layer, split_lengths, dim=1)
            value_parts = torch.split(mixed_value_layer, split_lengths, dim=1)

            key = None
            value = None
            outputs = []
            sum_length = 0
            for (query, _key, _value, part_length) in zip(query_parts, key_parts, value_parts, split_lengths):
                key = _key if key is None else torch.cat((key, _key), dim=1)
                value = _value if value is None else torch.cat((value, _value), dim=1)
                sum_length += part_length
                outputs.append(self.multi_head_attention(
                    query, key, value, attention_mask[:, :, sum_length - part_length: sum_length, :sum_length],
                    rel_pos=None if rel_pos is None else rel_pos[:, :, sum_length - part_length: sum_length,
                                                         :sum_length],
                )[0])
            outputs = (torch.cat(outputs, dim=1),)
        else:
            outputs = self.multi_head_attention(
                mixed_query_layer, mixed_key_layer, mixed_value_layer,
                attention_mask, rel_pos=rel_pos)
        return outputs


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None,
                split_lengths=None, rel_pos=None):
        self_outputs = self.self(
            hidden_states, attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            split_lengths=split_lengths, rel_pos=rel_pos)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, split_lengths=None, rel_pos=None):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask,
            split_lengths=split_lengths, rel_pos=rel_pos)
        attention_output = self_attention_outputs[0]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + self_attention_outputs[1:]
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.bus_num = config.bus_num
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])


    def forward(self, hidden_states, attention_mask, rel_pos=None, bus_connection=True):
        '''
        hidden_states: batch_size seq_length embed_dim
        attention_mask:
        rel_pos:
        Note: if use rel_pos, we need to consider the position of [CLS] in other segments
        '''
        all_hidden_states = ()
        all_attentions = ()

        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            new_hidden_states = ()

            if i == 0:
                if self.bus_num > 0 :
                    for seg_idx in range(self.bus_num):
                        layer_outputs = layer_module(
                            hidden_states[seg_idx], attention_mask[seg_idx][:,:,:,self.bus_num:],
                            rel_pos=rel_pos[seg_idx][:,:,self.bus_num:,self.bus_num:])

                        new_hidden_states = new_hidden_states + (layer_outputs[0],)

                else:
                    for seg_idx in range(len(hidden_states)):
                        layer_outputs = layer_module(
                            hidden_states[seg_idx], attention_mask[seg_idx],
                            rel_pos=rel_pos[seg_idx])
                        new_hidden_states = new_hidden_states + (layer_outputs[0],)

            else:
                if bus_connection:
                    station_emb = [hidden_states[idx][:,:1] for idx in  range(self.bus_num)]
                    station_emb = torch.cat(station_emb,1)

                    for seg_idx in range(self.bus_num):
                        seg_hidden_state = torch.cat([station_emb,hidden_states[seg_idx]],-2)
                        layer_outputs = layer_module(
                            seg_hidden_state, attention_mask[seg_idx],
                            rel_pos=rel_pos[seg_idx])

                        new_hidden_states = new_hidden_states + (layer_outputs[0][:,self.bus_num:],)

                else:
                    for seg_idx in range(len(hidden_states)):
                        layer_outputs = layer_module(
                            hidden_states[seg_idx], attention_mask[seg_idx],
                            rel_pos=rel_pos[seg_idx])
                        new_hidden_states = new_hidden_states + (layer_outputs[0],)

            hidden_states = new_hidden_states

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    """
    ret = 0
    if bidirectional:
        num_buckets //= 2
        # mtf.to_int32(mtf.less(n, 0)) * num_buckets
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance /
                                                        max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(
        val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret


class BusLM(TuringNLRv3PreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::
        input_ids = (seg1_ids,seg2_ids,seg3_ids)
        attention_mask = (seg1_mask, seg2_mask, seg3_mask)
        outputs = model(input_ids,attention_mask)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config):
        super(BusLM, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        if not isinstance(config, TuringNLRv3ForSeq2SeqConfig):
            self.pooler = BertPooler(config)
        else:
            self.pooler = None

        if self.config.rel_pos_bins > 0:
            self.rel_pos_bias = nn.Linear(self.config.rel_pos_bins+config.bus_num, config.num_attention_heads, bias=False)
        else:
            self.rel_pos_bias = None


    def forward(self, input_ids, attention_mask, seg_num=0, token_type_ids=None,
                position_ids=None, bus_connection=True):

        embedding_output, position_ids = self.embeddings(input_ids=input_ids)

        extended_attention_mask = ()
        for mask in attention_mask:
            mask = (1.0 - mask[:, None, None, :]) * -10000.0
            extended_attention_mask = extended_attention_mask + (mask,)

        if self.config.rel_pos_bins > 0:
            all_rel_pos = ()
            for seg_position in position_ids:
                rel_pos_mat = seg_position.unsqueeze(-2) - seg_position.unsqueeze(-1)
                rel_pos = relative_position_bucket(rel_pos_mat, num_buckets=self.config.rel_pos_bins, max_distance=self.config.max_rel_pos)

                if self.config.bus_num>0:
                    rel_pos = torch.cat([torch.zeros(rel_pos.size(0),self.config.bus_num,rel_pos.size(2)).to(rel_pos.device).long(),rel_pos],dim=1)
                    other_seg_relpos = torch.arange(self.config.rel_pos_bins,self.config.rel_pos_bins+self.config.bus_num).to(rel_pos.device).long()
                    other_seg_relpos = other_seg_relpos.unsqueeze(0).unsqueeze(0).expand(rel_pos.size(0),rel_pos.size(1),-1)
                    rel_pos = torch.cat([other_seg_relpos,rel_pos],dim=-1)

                rel_pos = F.one_hot(rel_pos, num_classes=self.config.rel_pos_bins+self.config.bus_num).type_as(embedding_output[0])
                rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
                all_rel_pos = all_rel_pos + (rel_pos,)
        else:
            all_rel_pos = None
            
        encoder_outputs = self.encoder(
            embedding_output, attention_mask=extended_attention_mask,
            rel_pos=all_rel_pos)

        return encoder_outputs   # last-layer hidden state, (all hidden states), (all attentions)


class BusLM_rec(TuringNLRv3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BusLM(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                position_ids=None):
        return self.bert(input_ids, attention_mask, token_type_ids=token_type_ids,position_ids=position_ids)


