import torch
import torch.nn as nn
from torchsummary import summary


class MyConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(MyConv3d, self).__init__()

        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv3d(x)
        return x


class MyFc(nn.Module):
    def __init__(self, in_channels, out_channels, drop=0.0):
        super(MyFc, self).__init__()

        self.fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class MyPool(nn.Module):

    def __init__(self):
        super(MyPool, self).__init__()

        self.down_avg = nn.AvgPool3d(kernel_size=(1, 8, 8), stride=1, padding=0)
        self.down_max = nn.MaxPool3d(kernel_size=(1, 8, 8), stride=1, padding=0)

    def forward(self, x):  # n c T 8 8
        u = self.down_avg(x)
        v = self.down_max(x)
        return torch.add(u, v)


class MyGRU(nn.Module):
    # input: n c T 1 1
    # output: n T c2

    def __init__(self, input_size, hidden_size, device, batch_first=True):
        # batch_first=True则输入输出的数据格式为 (batch, seq, feature)
        super(MyGRU, self).__init__()

        self.rnn = nn.GRU(input_size, hidden_size, batch_first=batch_first)
        self.pool = nn.AdaptiveMaxPool1d(1)

        self.hidden_size = hidden_size
        self.device = device

    def forward(self, x):  # n c T 1 1
        t = torch.squeeze(x, dim=3)
        t = torch.squeeze(t, dim=3)
        t = t.permute([0, 2, 1])

        r, h1 = self.rnn(t, self._get_initial_state(t.size(0), self.device))
        r = r.permute([0, 2, 1])
        f = self.pool(r).squeeze(2)
        return f

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0


# HSTVQA
class HSTVQA(nn.Module):

    def __init__(self, device):
        super(HSTVQA, self).__init__()
        self.device = device
        self._init_modules()

    def _init_modules(self):
        # f_in:8 f_out:8  256*256->64*64
        self.stage_0 = nn.Sequential(
            MyConv3d(3, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            MyConv3d(32, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            MyConv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            MyConv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
        )
        # f_in:8 f_out:4  64*64->32*32
        self.stage_1 = nn.Sequential(
            MyConv3d(64, 64, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
        )
        # f_in:4 f_out:2  32*32->16*16
        self.stage_2 = nn.Sequential(
            MyConv3d(64, 64, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
        )
        # f_in:2 f_out:1  16*16->8*8
        self.stage_3 = nn.Sequential(
            MyConv3d(64, 64, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            # nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
        )

        #######################################
        # side net
        # 8 64*64->8*8
        self.side_down_0 = nn.Sequential(
            MyConv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            MyConv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            MyConv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        # 4 32*32->8*8
        self.side_down_1 = nn.Sequential(
            MyConv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            MyConv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        # 2 16*16->8*8
        self.side_down_2 = nn.Sequential(
            MyConv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        # 1 8*8->8*8
        self.side_down_3 = nn.Sequential(
            MyConv3d(64, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        # UP
        self.side_pool_0 = MyPool()
        self.side_pool_1 = MyPool()
        self.side_pool_2 = MyPool()
        self.side_pool_3 = MyPool()

        # GRU
        self.side_reg_0 = MyGRU(64, 64, device=self.device, batch_first=True)
        self.side_reg_1 = MyGRU(64, 64, device=self.device, batch_first=True)
        self.side_reg_2 = MyGRU(64, 64, device=self.device, batch_first=True)

        ####################################
        # fc net, to get q
        self.fc_net_0 = MyFc(64, 1, drop=0.5)
        self.fc_net_1 = MyFc(64, 1, drop=0.5)
        self.fc_net_2 = MyFc(64, 1, drop=0.5)
        self.fc_net_3 = MyFc(64, 1, drop=0.5)

        self.att_net = nn.Sequential(
            nn.Linear(64 * 4, 64 * 4),
            nn.ReLU(inplace=True),
            nn.Linear(64 * 4, 64 * 4),
            nn.Sigmoid(),
        )

        self.fc_net_5 = nn.Sequential(
            MyFc(64 * 4, 1, drop=0.5),
        )

    def forward(self, x):
        x0 = self.stage_0(x)
        x1 = self.stage_1(x0)
        x2 = self.stage_2(x1)
        x3 = self.stage_3(x2)

        # stage0
        d0 = self.side_down_0(x0)
        p0 = self.side_pool_0(d0)
        r0 = self.side_reg_0(p0)
        t0 = torch.flatten(r0, 1)
        q0 = self.fc_net_0(t0).view(-1, 1)

        # stage1
        d1 = self.side_down_1(x1)
        p1 = self.side_pool_1(d1)
        r1 = self.side_reg_1(p1)
        t1 = torch.flatten(r1, 1)
        q1 = self.fc_net_1(t1).view(-1, 1)

        # stage2
        d2 = self.side_down_2(x2)
        p2 = self.side_pool_2(d2)
        r2 = self.side_reg_2(p2)
        t2 = torch.flatten(r2, 1)
        q2 = self.fc_net_2(t2).view(-1, 1)

        # stage3
        d3 = self.side_down_3(x3)
        p3 = self.side_pool_3(d3)
        t3 = torch.flatten(p3, 1)
        q3 = self.fc_net_3(t3).view(-1, 1)

        # last
        s = torch.cat((t0, t1, t2, t3), 1)
        sw = self.att_net(s)
        t = s.mul(sw)
        q = self.fc_net_5(t).view(-1, 1)

        # n * 5
        return torch.cat((q0, q1, q2, q3, q), 1)


class MyHuberLoss(torch.nn.Module):
    def __init__(self):
        super(MyHuberLoss, self).__init__()
        self.delta = 0.4

    # Pseudo-Huber loss
    def pseudo_huber_loss(self, target, pred, delta):
        return delta ** 2 * ((1 + ((pred - target) / delta) ** 2) ** 0.5 - 1)

    def compute_p(self, input):
        """
        :param input: [ q0, q1, q2, q3, q4, q, target ]
        :return: loss
        """
        pred = input[0:-2]
        final_pred = input[-2]
        target = input[-1]
        q_n = input.size()[-1] - 2

        loss_0 = self.pseudo_huber_loss(target, final_pred, self.delta)

        alpha = torch.zeros(q_n)
        alpha[:] = 0.3
        loss_a = [self.pseudo_huber_loss(target, pred[i], self.delta)
                  * alpha[i] for i in range(q_n)]
        loss_a = sum(loss_a) / q_n

        beta = torch.zeros(q_n)
        beta[:] = 0.3
        loss_b = [self.pseudo_huber_loss(final_pred, pred[i], self.delta)
                  * beta[i] for i in range(q_n)]
        loss_b = sum(loss_b) / q_n

        return loss_0 + loss_a + loss_b

    def forward(self, output, label):
        """
        :param output: [ n * [q0, q1, q2, q3, q4, q] ]
        :param label: [ n * 1 ]
        :return: avg loss
        """
        q_info = torch.cat((output, label), 1)
        loss = list(map(self.compute_p, q_info))
        loss = torch.stack(loss, 0).float()
        loss = torch.mean(loss)
        return loss


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    my_model = HSTVQA(device=device)
    my_model.to(device)
    summary(my_model, (3, 8, 256, 256))
