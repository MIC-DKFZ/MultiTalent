import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from multitalent.architectures.MedNeXt.blocks import *


class MedNeXt_for_MT(nn.Module):

    def __init__(self,
                 in_channels: int|dict,
                 n_channels: int,
                 n_classes: int|dict,
                 exp_r: int = 4,  # Expansion ratio as in Swin Transformers
                 kernel_size: int = 7,  # Ofcourse can test kernel_size
                 enc_kernel_size: int = None,
                 dec_kernel_size: int = None,
                 deep_supervision: bool = False,  # Can be used to test deep supervision
                 do_res: bool = False,  # Can be used to individually test residual connection
                 do_res_up_down: bool = False,  # Additional 'res' connection on up and down convs
                 checkpoint_style: bool = None,  # Either inside block or outside block
                 block_counts: list = [2, 2, 2, 2, 2, 2, 2, 2, 2],  # Can be used to test staging ratio:
                 # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
                 norm_type='group',
                 dim='3d',  # 2d or 3d
                 grn=False
                 ):

        super().__init__()

        self.do_ds = deep_supervision
        assert checkpoint_style in [None, 'outside_block']
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        assert dim in ['2d', '3d']

        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
        if isinstance(in_channels, int):
            self.stem = conv(in_channels, n_channels, kernel_size=1)
        elif isinstance(in_channels, dict):
            stem = {}
            for id in in_channels.keys():
                stem[id] = conv(in_channels[id], n_channels, kernel_size=1)
            self.stem = nn.ModuleDict(stem)
        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]

        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[0],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[0])]
                                         )

        self.down_0 = MedNeXtDownBlock(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim
        )

        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[1],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[1])]
                                         )

        self.down_1 = MedNeXtDownBlock(
            in_channels=2 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[2],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[2])]
                                         )

        self.down_2 = MedNeXtDownBlock(
            in_channels=4 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[3],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[3])]
                                         )

        self.down_3 = MedNeXtDownBlock(
            in_channels=8 * n_channels,
            out_channels=16 * n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 16,
                out_channels=n_channels * 16,
                exp_r=exp_r[4],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[4])]
                                        )

        self.up_3 = MedNeXtUpBlock(
            in_channels=16 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[5])]
                                         )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[6])]
                                         )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[7])]
                                         )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(block_counts[8])]
                                         )
        out_0 = {}
        if isinstance(n_classes, dict):
            for id in n_classes.keys():
                out_0[id] = OutBlock(in_channels=n_channels, n_classes=n_classes[id], dim=dim)
            self.out_0 = nn.ModuleDict(out_0)
        else:
            self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        if deep_supervision:
            if isinstance(n_classes, dict):
                out_1 = {}
                out_2 = {}
                out_3 = {}
                out_4 = {}
                for id in n_classes.keys():
                    out_1[id] = OutBlock(in_channels=n_channels * 2, n_classes=n_classes[id], dim=dim)
                    out_2[id] = OutBlock(in_channels=n_channels * 4, n_classes=n_classes[id], dim=dim)
                    out_3[id] = OutBlock(in_channels=n_channels * 8, n_classes=n_classes[id], dim=dim)
                    out_4[id] = OutBlock(in_channels=n_channels * 16, n_classes=n_classes[id], dim=dim)
                self.out_1 = nn.ModuleDict(out_1)
                self.out_2 = nn.ModuleDict(out_2)
                self.out_3 = nn.ModuleDict(out_3)
                self.out_4 = nn.ModuleDict(out_4)

            else:
                self.out_1 = OutBlock(in_channels=n_channels * 2, n_classes=n_classes, dim=dim)
                self.out_2 = OutBlock(in_channels=n_channels * 4, n_classes=n_classes, dim=dim)
                self.out_3 = OutBlock(in_channels=n_channels * 8, n_classes=n_classes, dim=dim)
                self.out_4 = OutBlock(in_channels=n_channels * 16, n_classes=n_classes, dim=dim)

        self.block_counts = block_counts

    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x

    def forward(self, x, ids=None):
        x_fromstem = None
        if type(self.stem) == nn.ModuleDict:
            for b, id in zip(x, ids):
                if x_fromstem is None:
                    x_fromstem = self.stem[id](b)
                else:
                    x_fromstem = torch.cat((x_fromstem, self.stem[id](b)), dim=0)

        else:
            for b in x:
                if x_fromstem is None:
                    x_fromstem = self.stem(b)
                else:
                    x_fromstem = torch.cat((x_fromstem, self.stem(b)), dim=0)
        x = x_fromstem
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor)
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor)
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor)
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor)

            x = self.iterative_checkpoint(self.bottleneck, x)
            #outer list for ids, inner fo ds
            seg_out = []
            if self.do_ds:
                for c,id in enumerate(ids):
                    if len(seg_out) == 0:
                        seg_out = [[checkpoint.checkpoint(self.out_4[id], torch.unsqueeze(x[c], dim=0), self.dummy_tensor)]]
                    else:
                        seg_out.append([checkpoint.checkpoint(self.out_4[id], torch.unsqueeze(x[c], dim=0), self.dummy_tensor)])
            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_3
            x = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if self.do_ds:
                for c,id in enumerate(ids):
                    seg_out[c].append(checkpoint.checkpoint(self.out_3[id], torch.unsqueeze(x[c], dim=0), self.dummy_tensor))
            del x_res_3, x_up_3

            x_up_2 = checkpoint.checkpoint(self.up_2, x, self.dummy_tensor)
            dec_x = x_res_2 + x_up_2
            x = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if self.do_ds:
                for c,id in enumerate(ids):
                    seg_out[c].append(checkpoint.checkpoint(self.out_2[id], torch.unsqueeze(x[c], dim=0), self.dummy_tensor))
            del x_res_2, x_up_2

            x_up_1 = checkpoint.checkpoint(self.up_1, x, self.dummy_tensor)
            dec_x = x_res_1 + x_up_1
            x = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if self.do_ds:
                for c,id in enumerate(ids):
                    seg_out[c].append(checkpoint.checkpoint(self.out_1[id], torch.unsqueeze(x[c], dim=0), self.dummy_tensor))
            del x_res_1, x_up_1

            x_up_0 = checkpoint.checkpoint(self.up_0, x, self.dummy_tensor)
            dec_x = x_res_0 + x_up_0
            x = self.iterative_checkpoint(self.dec_block_0, dec_x)
            del x_res_0, x_up_0, dec_x

            for c, id in enumerate(ids):
                if self.do_ds:
                    seg_out[c].append(checkpoint.checkpoint(self.out_0[id], torch.unsqueeze(x[c], dim=0), self.dummy_tensor))
                else:
                    if len(seg_out) == 0:
                        seg_out = [[checkpoint.checkpoint(self.out_0[id], torch.unsqueeze(x[c], dim=0), self.dummy_tensor)]]
                    else:
                        seg_out.append([checkpoint.checkpoint(self.out_0[id], torch.unsqueeze(x[c], dim=0), self.dummy_tensor)])

        else:
            print('should not reach this point for MT')
            x_res_0 = self.enc_block_0(x)
            x = self.down_0(x_res_0)
            x_res_1 = self.enc_block_1(x)
            x = self.down_1(x_res_1)
            x_res_2 = self.enc_block_2(x)
            x = self.down_2(x_res_2)
            x_res_3 = self.enc_block_3(x)
            x = self.down_3(x_res_3)

            x = self.bottleneck(x)
            if self.do_ds:
                x_ds_4 = self.out_4(x)

            x_up_3 = self.up_3(x)
            dec_x = x_res_3 + x_up_3
            x = self.dec_block_3(dec_x)

            if self.do_ds:
                x_ds_3 = self.out_3(x)
            del x_res_3, x_up_3

            x_up_2 = self.up_2(x)
            dec_x = x_res_2 + x_up_2
            x = self.dec_block_2(dec_x)
            if self.do_ds:
                x_ds_2 = self.out_2(x)
            del x_res_2, x_up_2

            x_up_1 = self.up_1(x)
            dec_x = x_res_1 + x_up_1
            x = self.dec_block_1(dec_x)
            if self.do_ds:
                x_ds_1 = self.out_1(x)
            del x_res_1, x_up_1

            x_up_0 = self.up_0(x)
            dec_x = x_res_0 + x_up_0
            x = self.dec_block_0(dec_x)
            del x_res_0, x_up_0, dec_x

            x = self.out_0(x)

        if self.do_ds:
            for c, id in enumerate(ids):
                seg_out[c] = seg_out[c][::-1]
            return seg_out
        else:
            return seg_out


if __name__ == "__main__":
    network = MedNeXt(
        in_channels=1,
        n_channels=32,
        n_classes=13,
        exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],  # Expansion ratio as in Swin Transformers
        # exp_r = 2,
        kernel_size=3,  # Can test kernel_size
        deep_supervision=True,  # Can be used to test deep supervision
        do_res=True,  # Can be used to individually test residual connection
        do_res_up_down=True,
        # block_counts = [2,2,2,2,2,2,2,2,2],
        block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3],
        checkpoint_style=None,
        dim='2d',
        grn=True

    ).cuda()


    # network = MedNeXt_RegularUpDown(
    #         in_channels = 1,
    #         n_channels = 32,
    #         n_classes = 13,
    #         exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
    #         kernel_size=3,                     # Can test kernel_size
    #         deep_supervision=True,             # Can be used to test deep supervision
    #         do_res=True,                      # Can be used to individually test residual connection
    #         block_counts = [2,2,2,2,2,2,2,2,2],
    #
    #     ).cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(count_parameters(network))

    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import parameter_count_table

    # model = ResTranUnet(img_size=128, in_channels=1, num_classes=14, dummy=False).cuda()
    x = torch.zeros((1, 1, 64, 64, 64), requires_grad=False).cuda()
    flops = FlopCountAnalysis(network, x)
    print(flops.total())

    with torch.no_grad():
        print(network)
        x = torch.zeros((1, 1, 128, 128, 128)).cuda()
        print(network(x)[0].shape)