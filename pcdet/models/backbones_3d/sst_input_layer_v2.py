# move the computation of position embeding and mask in middle_encoder_layer
import torch
from mmcv.runner import auto_fp16
from torch import nn
from pcdet.ops.sst.sst_ops import flat2window_v2, window2flat_v2, get_inner_win_inds, get_flat2win_inds_v2, get_window_coors



class SSTInputLayerV2(nn.Module):
    """
    This is one of the core class of SST, converting the output of voxel_encoder to sst input.
    There are 3 things to be done in this class:
    1. Reginal Grouping : assign window indices to each voxel.
    2. Voxel drop and region batching: see our paper for detail
    3. Pre-computing the transfomation information for converting flat features ([N x C]) to region features ([R, T, C]).
        R is the number of regions containing at most T tokens (voxels). See function flat2window and window2flat for details.

    Main args:
        drop_info (dict): drop configuration for region batching. 
        window_shape (tuple[int]): (num_x, num_y). Each window is divided to num_x * num_y pillars (including empty pillars).
        shift_list (list[tuple]): [(shift_x, shift_y), ]. shift_x = 5 means all windonws will be shifted for 5 voxels along positive direction of x-aixs.
        debug: apply strong assertion for developing. 
    """

    def __init__(self,
        drop_info,
        window_shape,
        sparse_shape,
        shuffle_voxels=True,
        debug=True,
        normalize_pos=False,
        pos_temperature=10000,
        mute=True,
        ):
        super().__init__()
        self.fp16_enabled = False
        self.meta_drop_info = drop_info
        self.sparse_shape = sparse_shape
        self.shuffle_voxels = shuffle_voxels
        self.debug = debug
        self.window_shape = window_shape
        self.normalize_pos = normalize_pos
        self.pos_temperature = pos_temperature
        self.mute = mute

        
    @auto_fp16(apply_to=('voxel_feat', ))
    def forward(self, voxel_feats, voxel_coors, top_mid_low_voxel_info=False):
        if top_mid_low_voxel_info:
            top_voxel_feats = voxel_feats[0]
            mid_low_voxel_feats = voxel_feats[1]
            
            top_voxel_info = self.get_voxel_info(top_voxel_feats, voxel_coors)
            mid_low_voxel_info = self.get_voxel_info(mid_low_voxel_feats, voxel_coors, cat=True)
            
            mid_low_voxel_info["mid_low_voxel_feats"] = mid_low_voxel_feats
            return [top_voxel_info, mid_low_voxel_info]
        else:
            if len(voxel_feats)==1:
                voxel_info = self.get_voxel_info(voxel_feats[0], voxel_coors)
                #voxel_info_2_window = self.get_voxel_info(voxel_feats[0], voxel_coors, multi_window='middle')
                #voxel_info_3_window = self.get_voxel_info(voxel_feats[0], voxel_coors, multi_window='high')
                #voxel_info = [voxel_info_1_window, voxel_info_2_window] # voxel_info_3_window]
                #self.window_shape = [int(x/2) for x in self.window_shape]
                #self.window_shape[-1] = 1
                return voxel_info
            
            voxel_info = self.get_voxel_info(voxel_feats[0], voxel_coors)
            voxel_cat_info= self.get_voxel_info(voxel_feats[0], voxel_coors, cat=True)
            if len(voxel_feats)==3:
                voxel_info["mid_voxel_feats"] = voxel_feats[1]
                voxel_info["low_voxel_feats"] = voxel_feats[2]
                return [voxel_info, voxel_cat_info]
            
            voxel_info["mid_low_voxel_feats"] = voxel_feats[1]
            return voxel_info
    
    def get_voxel_info(self, voxel_feats, voxel_coors, cat=False, multi_window=None):
        '''
        Args:
            voxel_feats: shape=[N, C], N is the voxel num in the batch.
            coors: shape=[N, 4], [b, z, y, x]
        Returns:
            feat_3d_dict: contains region features (feat_3d) of each region batching level. Shape of feat_3d is [num_windows, num_max_tokens, C].
            flat2win_inds_list: two dict containing transformation information for non-shifted grouping and shifted grouping, respectively. The two dicts are used in function flat2window and window2flat.
            voxel_info: dict containing extra information of each voxel for usage in the backbone.
        '''
    
        original_index = torch.arange(len(voxel_feats), device=voxel_feats.device)
        
        self.set_drop_info()
        
        #if multi_window=="high":
        #    self.shifts_list = [(int(x*1.5), int(y*1.5)) for x, y in  self.shifts_list]
        #    self.window_shape = [int(x*1.5) for x in self.window_shape]
        #    self.window_shape[-1] = 1
        #if multi_window=="middle":
        #   self.shifts_list = [(x*2, y*2) for x, y in  self.shifts_list]
        #    self.window_shape = [x*2 for x in self.window_shape]
        #    self.window_shape[-1] = 1
        voxel_coors = voxel_coors.long()

        if self.shuffle_voxels:
            # shuffle the voxels to make the drop process uniform.
            shuffle_inds = torch.randperm(len(voxel_feats))
            voxel_feats = voxel_feats[shuffle_inds]
            voxel_coors = voxel_coors[shuffle_inds]
            original_index = original_index[shuffle_inds]

        voxel_info = self.window_partition(voxel_coors) # Regional Grouping
        voxel_info['voxel_feats'] = voxel_feats
        voxel_info['voxel_coors'] = voxel_coors
        voxel_info['original_index'] = original_index
        voxel_info = self.drop_voxel(voxel_info, 2) # voxel_info is updated in this function 

        voxel_feats = voxel_info['voxel_feats']  # after dropping
        voxel_coors = voxel_info['voxel_coors']
        original_index = voxel_info['voxel_coors']

        for i in range(2):
            # Dict where for each drop level we give a index to each token
            # unique to all tokens in all windows of that drop level
            voxel_info[f'flat2win_inds_shift{i}'] = \
                get_flat2win_inds_v2(voxel_info[f'batch_win_inds_shift{i}'], voxel_info[f'voxel_drop_level_shift{i}'], self.drop_info, debug=True)

            # Same structure as above. Positional embedding is done using Sine-Cos embedding within a window, i.e.
            # not related to global position. Position in window is thus x_coord % windows_size_x and same for y
            # Sine-Cosine) 위치 임베딩을 계산
            if cat:
                voxel_info[f'pos_dict_shift{i}'] = \
                    self.get_pos_embed(voxel_info[f'flat2win_inds_shift{i}'], voxel_info[f'coors_in_win_shift{i}'], voxel_feats.size(1)*2, voxel_feats.dtype)
            else:
                voxel_info[f'pos_dict_shift{i}'] = \
                        self.get_pos_embed(voxel_info[f'flat2win_inds_shift{i}'], voxel_info[f'coors_in_win_shift{i}'], voxel_feats.size(1), voxel_feats.dtype)
            voxel_info[f'key_mask_shift{i}'] = \
                self.get_key_padding_mask(voxel_info[f'flat2win_inds_shift{i}'])
        if self.debug:
            coors_3d_dict_shift0 = flat2window_v2(voxel_coors, voxel_info['flat2win_inds_shift0'])
            coors_2d = window2flat_v2(coors_3d_dict_shift0, voxel_info['flat2win_inds_shift0'])
            assert (coors_2d == voxel_coors).all()

        if self.shuffle_voxels:
            voxel_info['shuffle_inds'] = shuffle_inds
                
        return voxel_info
         
    def drop_single_shift(self, batch_win_inds):
        drop_info = self.drop_info
        drop_lvl_per_voxel = -torch.ones_like(batch_win_inds)
        inner_win_inds = get_inner_win_inds(batch_win_inds)
        bincount = torch.bincount(batch_win_inds)
        num_per_voxel_before_drop = bincount[batch_win_inds] #
        target_num_per_voxel = torch.zeros_like(batch_win_inds)

        for dl, v in enumerate(drop_info):
            max_tokens = drop_info[dl]['max_tokens']
            lower, upper = drop_info[dl]['drop_range']
            range_mask = (num_per_voxel_before_drop >= lower) & (num_per_voxel_before_drop < upper)
            target_num_per_voxel[range_mask] = max_tokens
            drop_lvl_per_voxel[range_mask] = dl
        
        if self.debug:
            assert (target_num_per_voxel > 0).all()
            assert (drop_lvl_per_voxel >= 0).all()
        
        keep_mask = inner_win_inds < target_num_per_voxel
        assert (keep_mask == True).all()
        
        return keep_mask, drop_lvl_per_voxel

    def drop_voxel(self, voxel_info, num_shifts):
        '''
        To make it clear and easy to follow, we do not use loop to process two shifts.

        Separates windows by the number of tokens e.g.:
        group 1: 0-30 tokens, pad windows with less than 30 token to 30 token
        group 2: 30-60 tokens, pad windows with less than 60 tokens to 60 tokens
        group 3: 60-100000 tokens, pad windows with less than 100 token to 100 tokens
                    drop tokens in windows with more than 100 tokens so that they have 100 tokens
        '''

        batch_win_inds_s0 = voxel_info['batch_win_inds_shift0']
        num_all_voxel = batch_win_inds_s0.shape[0]

        voxel_keep_inds = torch.arange(num_all_voxel, device=batch_win_inds_s0.device, dtype=torch.long)

        keep_mask_s0, drop_lvl_s0 = self.drop_single_shift(batch_win_inds_s0)
        if self.debug:
            assert (drop_lvl_s0 >= 0).all()

        drop_lvl_s0 = drop_lvl_s0[keep_mask_s0]
        voxel_keep_inds = voxel_keep_inds[keep_mask_s0]
        batch_win_inds_s0 = batch_win_inds_s0[keep_mask_s0]

        if num_shifts == 1:
            voxel_info['voxel_keep_inds'] = voxel_keep_inds
            voxel_info['voxel_drop_level_shift0'] = drop_lvl_s0
            voxel_info['batch_win_inds_shift0'] = batch_win_inds_s0
            return voxel_info

        batch_win_inds_s1 = voxel_info['batch_win_inds_shift1']
        batch_win_inds_s1 = batch_win_inds_s1[keep_mask_s0]

        keep_mask_s1, drop_lvl_s1 = self.drop_single_shift(batch_win_inds_s1)
        if self.debug:
            assert (drop_lvl_s1 >= 0).all()

        # drop data in first shift again
        drop_lvl_s0 = drop_lvl_s0[keep_mask_s1]
        voxel_keep_inds = voxel_keep_inds[keep_mask_s1]
        batch_win_inds_s0 = batch_win_inds_s0[keep_mask_s1]

        drop_lvl_s1 = drop_lvl_s1[keep_mask_s1]
        batch_win_inds_s1 = batch_win_inds_s1[keep_mask_s1]

        voxel_info['voxel_keep_inds'] = voxel_keep_inds
        voxel_info['voxel_drop_level_shift0'] = drop_lvl_s0
        voxel_info['batch_win_inds_shift0'] = batch_win_inds_s0
        voxel_info['voxel_drop_level_shift1'] = drop_lvl_s1
        voxel_info['batch_win_inds_shift1'] = batch_win_inds_s1
        voxel_keep_inds = voxel_info['voxel_keep_inds']

        voxel_num_before_drop = len(voxel_info['voxel_coors'])
        voxel_info['voxel_feats'] = voxel_info['voxel_feats'][voxel_keep_inds]
        voxel_info['voxel_coors'] = voxel_info['voxel_coors'][voxel_keep_inds]
        voxel_info['original_index'] = voxel_info['original_index'][voxel_keep_inds]

        # Some other variables need to be dropped.
        for k, v in voxel_info.items():
            if isinstance(v, torch.Tensor) and len(v) == voxel_num_before_drop:
                voxel_info[k] = v[voxel_keep_inds]

        ### sanity check
        if self.debug and self.training:
            for dl, v in enumerate(self.drop_info):
                max_tokens = self.drop_info[dl]['max_tokens']

                mask_s0 = drop_lvl_s0 == dl
                if not mask_s0.any():
                    if not self.mute:
                        print(f'No voxel belongs to drop_level:{dl} in shift 0')
                    continue
                real_max = torch.bincount(batch_win_inds_s0[mask_s0]).max()
                assert real_max <= max_tokens, f'real_max({real_max}) > {max_tokens} in shift0'

                mask_s1 = drop_lvl_s1 == dl
                if not mask_s1.any():
                    if not self.mute:
                        print(f'No voxel belongs to drop_level:{dl} in shift 1')
                    continue
                real_max = torch.bincount(batch_win_inds_s1[mask_s1]).max()
                assert real_max <= max_tokens, f'real_max({real_max}) > {max_tokens} in shift1'
        ###
        return voxel_info

    @torch.no_grad()
    def window_partition(self, coors):
        voxel_info = {}
        for i in range(2):
            # Adds indexation which window (counted across all batches) and which spot in the window
            # 복셀 좌표를 윈도우 단위로 분할하고 인덱스화하여, 모델이 윈도우 내에서 작업을 수행하고 윈도우 간의 연산을 처리
            batch_win_inds, coors_in_win = get_window_coors(coors, self.sparse_shape, self.window_shape, i == 1)
            voxel_info[f'batch_win_inds_shift{i}'] = batch_win_inds # batch_win_inds는 각 voxel에 대한 인덱스
            voxel_info[f'coors_in_win_shift{i}'] = coors_in_win # coors_in_win은 각 복셀의 윈도우 내에서의 상대적인 위치 좌표 정보
        
        return voxel_info

    @torch.no_grad()
    def get_pos_embed(self, inds_dict, coors_in_win, feat_dim, dtype):
        '''
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        '''

        # [N,]
        window_shape = self.window_shape
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif  window_shape[-1] == 1:
            ndim = 2
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        assert coors_in_win.size(1) == 3
        z, y, x = coors_in_win[:, 0] - win_z/2, coors_in_win[:, 1] - win_y/2, coors_in_win[:, 2] - win_x/2
        assert (x >= -win_x/2 - 1e-4).all()
        assert (x <= win_x/2-1 + 1e-4).all()

        if self.normalize_pos:
            x = x / win_x * 2 * 3.1415 #[-pi, pi]
            y = y / win_y * 2 * 3.1415 #[-pi, pi]
            z = z / win_z * 2 * 3.1415 #[-pi, pi]
        
        pos_length = feat_dim // ndim
        # [pos_length]
        inv_freq = torch.arange(
            pos_length, dtype=torch.float32, device=coors_in_win.device)
        inv_freq = self.pos_temperature ** (2 * (inv_freq // 2) / pos_length)

        # [num_tokens, pos_length]
        embed_x = x[:, None] / inv_freq[None, :]
        embed_y = y[:, None] / inv_freq[None, :]
        if ndim == 3:
            embed_z = z[:, None] / inv_freq[None, :]

        # [num_tokens, pos_length]
        embed_x = torch.stack([embed_x[:, ::2].sin(), embed_x[:, 1::2].cos()], dim=-1).flatten(1)
        embed_y = torch.stack([embed_y[:, ::2].sin(), embed_y[:, 1::2].cos()], dim=-1).flatten(1)
        if ndim == 3:
            embed_z = torch.stack([embed_z[:, ::2].sin(), embed_z[:, 1::2].cos()], dim=-1).flatten(1)

        # [num_tokens, c]
        if ndim == 3:
            pos_embed_2d = torch.cat([embed_x, embed_y, embed_z], dim=-1).to(dtype)
        else:
            pos_embed_2d = torch.cat([embed_x, embed_y], dim=-1).to(dtype)
        
        gap = feat_dim - pos_embed_2d.size(1)
        assert gap >= 0
        if gap > 0:
            assert ndim == 3
            padding = torch.zeros((pos_embed_2d.size(0), gap), dtype=dtype, device=coors_in_win.device)
            pos_embed_2d = torch.cat([pos_embed_2d, padding], dim=1)
        else:
            assert ndim == 2

        pos_embed_dict = flat2window_v2(
            pos_embed_2d, inds_dict)

        return pos_embed_dict

    @torch.no_grad()
    def get_key_padding_mask(self, ind_dict):
        num_all_voxel = len(ind_dict['voxel_drop_level'])
        key_padding = torch.ones((num_all_voxel, 1)).to(ind_dict['voxel_drop_level'].device).bool()

        window_key_padding_dict = flat2window_v2(key_padding, ind_dict)

        # logical not. True means masked
        for key, value in window_key_padding_dict.items():
            window_key_padding_dict[key] = value.logical_not().squeeze(2)
        
        return window_key_padding_dict

    def set_drop_info(self):
        if hasattr(self, 'drop_info'):
            return
        meta = self.meta_drop_info
        if isinstance(meta, tuple):
            if self.training:
                self.drop_info = meta[0]
            else:
                self.drop_info = meta[1]
        else:
            self.drop_info = meta
        print(f'drop_info is set to {self.drop_info}, in input_layer')
        