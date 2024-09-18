import copy
import warnings

import torch
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, BaseTransformerLayer
from mmdet.models import inverse_sigmoid


class XjBevDecoder(TransformerLayerSequence):

    def forward(self,
                query,
                key,
                value,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                reg_branches=None,
                reference_points=None,
                return_intermediate=True,
                **kwargs):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            output = layer(
                output,
                key,
                value,
                query_pos,
                key_pos,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                reference_points=reference_points_input,
                **kwargs
            )
            output = output.permute(1, 0, 2)
            tmp = reg_branches[lid](output)
            inver_ref=inverse_sigmoid(reference_points)#torch.float16 ->torch.float32
            reference_points = tmp + inver_ref
            reference_points = reference_points.sigmoid()
            output = output.permute(1, 0, 2)
            if return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
            reference_points = reference_points.detach()
        if return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points




class TransformerDecoderLayer(BaseTransformerLayer):

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                self_attn_mask=None,
                num_vec=50,
                num_pts_per_vec=20,
                **kwargs) -> torch.Tensor:
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                                                     f'operation_order {self.num_attn}'
        for layer in self.operation_order:
            if layer == 'self_attn':
                if attn_index == 0:
                    n_pts, n_batch, n_dim = query.shape
                    query = query.view(num_vec, num_pts_per_vec, n_batch, n_dim).flatten(1, 2)
                    query_pos = query_pos.view(num_vec, num_pts_per_vec, n_batch, n_dim).flatten(1, 2)
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=self_attn_mask,
                        key_padding_mask=query_key_padding_mask,
                        **kwargs
                    )
                    query = query.view(num_vec, num_pts_per_vec, n_batch, n_dim).flatten(0, 1)
                    query_pos = query_pos.view(num_vec, num_pts_per_vec, n_batch, n_dim).flatten(0, 1)
                    attn_index += 1
                    identity = query
                else:
                    n_pts, n_batch, n_dim = query.shape
                    query = query.view(num_vec,
                                       num_pts_per_vec,
                                       n_batch,
                                       n_dim).permute(1, 0, 2, 3).contiguous().flatten(1, 2)
                    query_pos = query_pos.view(num_vec,
                                               num_pts_per_vec,
                                               n_batch,
                                               n_dim).permute(1, 0, 2, 3).contiguous().flatten(1, 2)
                    temp_key = temp_value = query
                    query = self.attentions[attn_index](
                        query,
                        temp_key,
                        temp_value,
                        identity if self.pre_norm else None,
                        query_pos=query_pos,
                        key_pos=query_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=query_key_padding_mask,
                        **kwargs
                    )
                    query = query.view(num_pts_per_vec,
                                       num_vec,
                                       n_batch,
                                       n_dim).permute(1, 0, 2, 3).contiguous().flatten(0, 1)
                    query_pos = query_pos.view(num_pts_per_vec,
                                               num_vec,
                                               n_batch,
                                               n_dim).permute(1, 0, 2, 3).contiguous().flatten(0, 1)
                    attn_index += 1
                    identity = query
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs
                )
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
