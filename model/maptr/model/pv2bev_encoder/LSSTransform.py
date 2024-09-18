import torch
from mmdet.models.backbones.resnet import BasicBlock
from mmengine.model import BaseModule
from torch import nn
from torch.amp import autocast

from utils import bev_pool, bev_pool_v2


def gen_dx_bx(x_bound, y_bound, z_bound):
    dx = torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]])
    bx = torch.Tensor([row[0] for row in [x_bound, y_bound, z_bound]])
    nx = torch.Tensor(
        [int((row[1] - row[0]) / row[2]) for row in [x_bound, y_bound, z_bound]]
    )
    return dx, bx, nx


class LSSTransform(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pc_range,
                 voxel_size,
                 d_bound,
                 down_sample: int = 1):
        super().__init__()
        self.d_bound = d_bound
        self.out_channels = out_channels
        self.depth = int((d_bound[1] - d_bound[0]) / d_bound[2])
        self.depth_net = DepthNet(
            in_channels, in_channels, self.out_channels, self.depth
        )
        self.x_bound = [pc_range[0], pc_range[3], voxel_size[0]]
        self.y_bound = [pc_range[1], pc_range[4], voxel_size[1]]
        self.z_bound = [pc_range[2], pc_range[5], voxel_size[2]]
        dx, bx, nx = gen_dx_bx(self.x_bound, self.y_bound, self.z_bound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)
        self.d_s = down_sample
        if self.d_s > 1:
            self.down_sample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, stride=self.d_s, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image: torch.Tensor, img_metas, only_bev=False):
        b, n, c, fh, fw = image.shape
        device = image.device
        ih = img_metas[0].img_shape[0]
        iw = img_metas[0].img_shape[1]
        # 生成img下的深度点
        frustum = self.create_frustum(fh, fw, ih, iw).to(device)
        lidar2img = []
        img_aug_matrix = []
        cam_intrinsics = []
        for meta in img_metas:
            lidar2img.append(meta.lidar2img)
            img_aug_matrix.append(meta.img_aug_matrix)
            cam_intrinsics.append(meta.cam_intrinsics)
        lidar2img = torch.stack(lidar2img).to(torch.float32).to(device)
        img_aug_matrix = torch.stack(img_aug_matrix).to(torch.float32).to(device)
        cam_intrinsics = torch.stack(cam_intrinsics).to(torch.float32).to(device)
        # 将img下的深度点转移到bev下
        frustum[..., :2] = frustum[..., :2] * frustum[..., 2:3]
        img2lidar = torch.inverse(lidar2img)
        point = (img2lidar[..., :3, :3].reshape(b, n, 1, 1, 1, 3, 3).matmul(frustum.unsqueeze(-1)) -
                 img2lidar[..., :3, 3].reshape(b, n, 1, 1, 1, 3, 1))
        point = point.squeeze(-1)
        # 得到mlp的输入
        sensor2lidar = img2lidar.matmul(img_aug_matrix).matmul(cam_intrinsics)
        mlp_input = torch.stack([
            cam_intrinsics[..., 0, 0],
            cam_intrinsics[..., 1, 1],
            cam_intrinsics[..., 0, 2],
            cam_intrinsics[..., 1, 2],
            img_aug_matrix[..., 0, 0],
            img_aug_matrix[..., 0, 1],
            img_aug_matrix[..., 0, 3],
            img_aug_matrix[..., 1, 0],
            img_aug_matrix[..., 1, 1],
            img_aug_matrix[..., 1, 3]
        ], dim=-1)
        mlp_input = torch.cat([
            mlp_input,
            sensor2lidar[:, :, :3, :].reshape(b, n, -1)
        ], dim=-1)
        # 开始计算
        image = image.reshape(b * n, c, fh, fw)
        bev_feat = self.depth_net(image, mlp_input)#6 324 15 25
        # bev_pool_v1
        depth = bev_feat[:, :self.depth].softmax(dim=1)#6 68 15 25
        bev_feat = depth.unsqueeze(1) * bev_feat[:, self.depth:self.depth + self.out_channels, ...].unsqueeze(2)
        bev_feat = bev_feat.view(b, n, self.out_channels, self.depth, fh, fw)
        bev_feat = bev_feat.permute(0, 1, 3, 4, 5, 2)
        depth = depth.view(b, n, self.depth, fh, fw)
        bev_feat = self.bev_pool(point, bev_feat)
        # 将maptrv2中使用的bev_pool_v1修改为v2版
        # depth = x[:, :self.depth].softmax(dim=1)
        # tran_feat = x[:, self.depth:self.depth + self.out_channels, ...]
        # depth = depth.reshape(b, n, self.depth, fh, fw)
        # tran_feat = tran_feat.reshape(b, n, self.out_channels, fh, fw)
        # bev_feat = self.bev_pool_v2(point, depth, tran_feat)
        bev_feat = bev_feat.permute(0, 1, 3, 2).contiguous()
        if self.d_s > 1:
            bev_feat = self.down_sample(bev_feat)
        bev_feat = bev_feat.reshape(b, c, -1).permute(0, 2, 1).contiguous()
        if only_bev:
            return bev_feat
        return bev_feat, depth

    def create_frustum(self, fh, fw, ih, iw):
        ds = torch.arange(*self.d_bound, dtype=torch.float).view(-1, 1, 1).expand(-1, fh, fw)
        d, _, _ = ds.shape
        xs = torch.linspace(0, iw - 1, fw, dtype=torch.float).view(1, 1, fw).expand(d, fh, fw)
        ys = torch.linspace(0, ih - 1, fh, dtype=torch.float).view(1, fh, 1).expand(d, fh, fw)
        frustum = torch.stack((xs, ys, ds), -1)
        return frustum

    def bev_pool(self, point: torch.Tensor, x: torch.Tensor):
        b, n, d, h, w, c = x.shape#1 6 68 15 25
        prime = b * n * d * h * w
        device = x.device
        x = x.reshape(prime, c)
        point = ((point - self.bx) / self.dx).long()
        point = point.reshape(prime, 3)#153000 3
        batch_ix = torch.cat([
            torch.full([prime // b, 1], ix, device=device, dtype=torch.long)
            for ix in range(b)
        ])
        point = torch.cat((point, batch_ix), 1)#153000 4
        kept = (
                (point[:, 0] >= 0) & (point[:, 0] < self.nx[0])
                & (point[:, 1] >= 0) & (point[:, 1] < self.nx[1])
                & (point[:, 2] >= 0) & (point[:, 2] < self.nx[2])
        )
        x = x[kept]#74265 256
        point = point[kept]
        if x.is_cpu:
            x = bev_pool(x.cuda(), point.cuda(), b, self.nx[2], self.nx[0], self.nx[1]).cpu()
        else:
            x = bev_pool(x, point, b, self.nx[2], self.nx[0], self.nx[1])#1 256 1 200 400
        x = torch.cat(x.unbind(dim=2), 1)#1 256 200 400
        return x

    def bev_pool_v2(self, point, depth, tran_feat):
        b, n, d, h, w, c = point.shape
        device = point.device
        prime = b * n * d * h * w
        ranks_depth = torch.arange(
            0, prime, dtype=torch.int, device=device)
        ranks_feat = torch.arange(
            0, prime // d, dtype=torch.int, device=device)
        ranks_feat = ranks_feat.reshape(b, n, 1, h, w)
        ranks_feat = ranks_feat.expand(b, n, d, h, w).flatten()
        point = ((point - self.bx) / self.dx).long()
        point = point.reshape(prime, 3)
        batch_ix = torch.cat([
            torch.full([prime // b, 1], ix, device=device, dtype=torch.long)
            for ix in range(b)
        ])
        point = torch.cat((point, batch_ix), 1)
        kept = (
                (point[:, 0] >= 0) & (point[:, 0] < self.nx[0])
                & (point[:, 1] >= 0) & (point[:, 1] < self.nx[1])
                & (point[:, 2] >= 0) & (point[:, 2] < self.nx[2])
        )
        if len(kept) == 0:
            dummy = torch.zeros(size=[
                tran_feat.shape[0], tran_feat.shape[2],
                int(self.nx[2]),
                int(self.nx[0]),
                int(self.nx[1])
            ]).to(tran_feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        point, ranks_depth, ranks_feat = point[kept], ranks_depth[kept], ranks_feat[kept]
        ranks_bev = point[:, 3] * (self.nx[2] * self.nx[1] * self.nx[0])
        ranks_bev += point[:, 2] * (self.nx[1] * self.nx[0])
        ranks_bev += point[:, 1] * self.nx[0] + point[:, 0]
        kept = torch.ones(ranks_bev.shape[0], device=device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            dummy = torch.zeros(size=[
                tran_feat.shape[0], tran_feat.shape[2],
                int(self.nx[2]),
                int(self.nx[0]),
                int(self.nx[1])
            ]).to(tran_feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        tran_feat = tran_feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (b, int(self.nx[2]), int(self.nx[1]), int(self.nx[0]), self.out_channels)
        if point.is_cpu:
            bev_feat = bev_pool_v2(depth.cuda(), tran_feat.cuda(), ranks_depth.cuda(), ranks_feat.cuda(),
                                   ranks_bev.cuda(), bev_feat_shape, interval_starts.cuda(),
                                   interval_lengths.cuda()).cpu()
        else:
            bev_feat = bev_pool_v2(depth, tran_feat, ranks_depth, ranks_feat, ranks_bev,
                                   bev_feat_shape, interval_starts, interval_lengths)
        bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat

    def get_depth_loss(self, depth_gt: torch.Tensor, depth: torch.Tensor):
        if depth is None:
            return 0
        b, n, h, w = depth_gt.shape
        depth_gt = depth_gt.reshape(b * n, h // 32, 32, w // 32, 32)
        depth_gt = depth_gt.permute(0, 1, 3, 2, 4).reshape(-1, 32 * 32).contiguous()
        tmp = torch.where(depth_gt == 0, torch.full_like(depth_gt, 1e5), depth_gt)
        depth_gt = torch.min(tmp, dim=-1).values
        depth_gt = depth_gt.reshape(b * n, h // 32, w // 32)
        depth_gt = (depth_gt - self.d_bound[0]) / self.d_bound[2]
        depth_gt = torch.where((depth_gt <= self.depth) & (depth_gt > 0),
                               depth_gt,
                               torch.zeros_like(depth_gt))
        depth_gt = nn.functional.one_hot(depth_gt.long(), self.depth).reshape(-1, self.depth).float()
        depth = depth.permute(0, 1, 3, 4, 2).reshape(-1, self.depth)
        mask = depth_gt > 0
        if mask.sum() == 0:
            return torch.tensor(0, device=depth.device)
        depth_gt = depth_gt[mask]
        depth = depth[mask]
        # 取消混合精度提升loss精度
        with autocast(enabled=False, device_type=depth.device.type):
            depth_loss = nn.functional.binary_cross_entropy(
                depth,
                depth_gt
            )
        return torch.nan_to_num(depth_loss)


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer(inplace=True)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(
            int(mid_channels * 5), inplanes, 1, bias=False)
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = nn.functional.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DepthNet(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 aspp_mid_channels=96,
                 only_depth=False):
        super().__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.only_depth = only_depth or context_channels == 0
        if not self.only_depth:
            self.context_conv = nn.Conv2d(
                mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
            self.context_mlp = Mlp(22, mid_channels, mid_channels)
            self.context_se = SELayer(mid_channels)
        self.bn = nn.BatchNorm1d(22)
        self.depth_mlp = Mlp(22, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)
        if aspp_mid_channels < 0:
            aspp_mid_channels = mid_channels
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, aspp_mid_channels),
            nn.Conv2d(mid_channels, depth_channels, 1, 1, 0)
        )

    def forward(self, x, mlp_input):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        if not self.only_depth:
            context_se = self.context_mlp(mlp_input)[..., None, None]
            context = self.context_se(x, context_se)
            context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        if not self.only_depth:
            return torch.cat([depth, context], dim=1)
        else:
            return depth
