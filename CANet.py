import math
import torch
import torchvision
import numpy as np
import torch.nn as nn
from torch.nn import init

class vgg(nn.Module):
    
    def __init__(self, pretrain=None, logger=None):
        super(vgg, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, (3, 3), stride=1, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
 
        self.conv2_1 = nn.Conv2d(64, 128, (3, 3), stride=1, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, (3, 3), stride=1, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3_1 = nn.Conv2d(128, 256, (3, 3), stride=1, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, (3, 3), stride=1, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)

        self.conv5_1 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_3 = nn.ReLU(inplace=True)

        if pretrain:
            if '.npy' in pretrain:
                state_dict = np.load(pretrain).item()
                for k in state_dict:
                    state_dict[k] = torch.from_numpy(state_dict[k])
            else:
                state_dict = torch.load(pretrain)
            own_state_dict = self.state_dict()
            for name, param in own_state_dict.items():
                if name in state_dict:
                    if logger:
                        logger.info('copy the weights of %s from pretrained model' % name)
                    param.copy_(state_dict[name])
                else:
                    if logger:
                        logger.info('init the weights of %s from mean 0, std 0.01 gaussian distribution'\
                         % name)
                    if 'bias' in name:
                        param.zero_()
                    else:
                        param.normal_(0, 0.01)
        else:
            self._initialize_weights(logger)

    def forward(self, x):
        conv1_1 = self.relu1_1(self.conv1_1(x))            # 256*256
        conv1_2 = self.relu1_2(self.conv1_2(conv1_1))
        pool1 = self.pool1(conv1_2)                        # pool1 torch.Size([1, 64, 128, 128])
        conv2_1 = self.relu2_1(self.conv2_1(pool1))        # 128*128
        conv2_2 = self.relu2_2(self.conv2_2(conv2_1))
        pool2 = self.pool2(conv2_2)                        # pool2 torch.Size([1, 128, 64, 64])
        conv3_1 = self.relu3_1(self.conv3_1(pool2))        # 64*64
        conv3_2 = self.relu3_2(self.conv3_2(conv3_1))
        conv3_3 = self.relu3_3(self.conv3_3(conv3_2))      
        pool3 = self.pool3(conv3_3)                        # pool3 torch.Size([1, 256, 32, 32])
        conv4_1 = self.relu4_1(self.conv4_1(pool3))        # 32*32
        conv4_2 = self.relu4_2(self.conv4_2(conv4_1))
        conv4_3 = self.relu4_3(self.conv4_3(conv4_2))
        pool4 = self.pool4(conv4_3)                        # pool4 torch.Size([1, 512, 31, 31])
        conv5_1 = self.relu5_1(self.conv5_1(pool4))
        conv5_2 = self.relu5_2(self.conv5_2(conv5_1))
        conv5_3 = self.relu5_3(self.conv5_3(conv5_2)) 
        side = [
                conv1_1, conv1_2, 
                conv2_1, conv2_2,
                conv3_1, conv3_2, conv3_3, 
                conv4_1, conv4_2, conv4_3, 
                conv5_1, conv5_2, conv5_3,
               ]

        return side

    def _initialize_weights(self, logger=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if logger:
                        logger.info('init the weights of %s from mean 0, std 0.01 gaussian distribution'\
                         % m)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class PAM_Module(nn.Module):

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma      = nn.Parameter(torch.zeros(1))
        self.softmax    = nn.Softmax(dim=-1)
    def forward(self, x):
       
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key   = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy     = torch.bmm(proj_query, proj_key)
        attention  = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out        = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out        = out.view(m_batchsize, C, height, width)
        out        = self.gamma * out * x

        return out


class R_PAM_Module(nn.Module):

    def __init__(self, in_dim):
        super(R_PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma      = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
      
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key   = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy     = torch.bmm(proj_query, proj_key)
        attention  = 1-torch.sigmoid(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out        = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out        = out.view(m_batchsize, C, height, width)
        out        = self.gamma * out * x

        return out    

def crop(data1, data2, crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    _, _, h2, w2 = data2.size()
    #assert(h2 <= h1 and w2 <= w1)
    data = data1[:, :, crop_h:crop_h+h2, crop_w:crop_w+w2]
    return data

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    #Make a 2D bilinear kernel suitable for upsampling
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class MDBlock(nn.Module):
    def __init__(self, c_in, rate=1):
        super(MDBlock, self).__init__()
        c_out = c_in
        self.rate = rate
        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        dilation = self.rate*1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        dilation = self.rate*3 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        dilation = self.rate*5 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)
        dilation = self.rate*7 if self.rate >= 1 else 1
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu4 = nn.ReLU(inplace=True)
        self._initialize_weights()
    def forward(self, x):
        o  = self.relu(self.conv(x))
        o1 = self.relu1(self.conv1(o))
        o2 = self.relu2(self.conv2(o))
        o3 = self.relu3(self.conv3(o))
        o4 = self.relu4(self.conv4(o))
        out = o + o1 + o2 + o3 + o4
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()  


class CANet(nn.Module):
    def __init__(self, pretrain=None, logger=None ,rate=1):   # 4
        super(CANet, self).__init__()
        t = 1
        self.pretrain = pretrain
        self.features     = vgg(pretrain, logger)
        self.msblock1_1   = MDBlock(64, rate)
        self.msblock1_2   = MDBlock(64, rate)
        self.conv1_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv1_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn1   = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn1_1 = nn.Conv2d(21, 1, 1, stride=1)
        self.msblock2_1   = MDBlock(128, rate)
        self.msblock2_2   = MDBlock(128, rate)
        self.conv2_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv2_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn2   = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn2_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock3_1   = MDBlock(256, rate)
        self.msblock3_2   = MDBlock(256, rate)
        self.msblock3_3   = MDBlock(256, rate)
        self.conv3_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn3   = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn3_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock4_1   = MDBlock(512, rate)
        self.msblock4_2   = MDBlock(512, rate)
        self.msblock4_3   = MDBlock(512, rate)
        self.conv4_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv4_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv4_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn4   = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn4_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock5_1   = MDBlock(512, rate)
        self.msblock5_2   = MDBlock(512, rate)
        self.msblock5_3   = MDBlock(512, rate)
        self.conv5_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv5_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv5_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn5   = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn5_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
       
        self.upsample_2   = nn.ConvTranspose2d(1, 1, 4, stride=2, bias=False)
        self.upsample_4   = nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        self.upsample_8   = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.upsample_8_5 = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.upsample_8_5_pam  = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.upsample_8_5_nam  = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.upsample_8_5_pana = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)

        self.pam_conv1 = nn.Conv2d(512, 256, 3, padding=1, bias=False)
        self.pam_relu1 = nn.PReLU()
        self.pam       = PAM_Module(256)
        self.pam_conv2 = nn.Conv2d(256, 128, 3, padding=1, bias=False)
        self.pam_relu2 = nn.PReLU()
        self.pam_drop  = nn.Dropout2d(0.1, False)
        self.pam_conv3 = nn.Conv2d(128, 1, 1)
        self.pam_relu3 = nn.PReLU()
        
        self.nam_conv1 = nn.Conv2d(512, 256, 3, padding=1, bias=False)
        self.nam_relu1 = nn.PReLU()
        self.nam       = R_PAM_Module(256)
        self.nam_conv2 = nn.Conv2d(256, 128, 3, padding=1, bias=False)
        self.nam_relu2 = nn.PReLU()
        self.nam_drop  = nn.Dropout2d(0.1, False)
        self.nam_conv3 = nn.Conv2d(128, 1, 1)
        self.nam_relu3 = nn.PReLU()
        
        self.pana_conv1 = nn.Conv2d(128, 1, 1)
        self.pana_drop  = nn.Dropout2d(0.1, False)
        self.pana_relu1 = nn.PReLU()

        self.relu_sum1 = nn.PReLU()
        self.relu_sum2 = nn.PReLU()
        self.relu_sum3 = nn.PReLU()
        self.relu_sum4 = nn.PReLU()
        self.relu_sum5 = nn.PReLU()
        
        self.relu_s1  = nn.PReLU()
        self.relu_s11 = nn.PReLU()
        self.relu_s2  = nn.PReLU()
        self.relu_s21 = nn.PReLU()
        self.relu_s3  = nn.PReLU()
        self.relu_s31 = nn.PReLU()
        self.relu_s4  = nn.PReLU()
        self.relu_s41 = nn.PReLU()
        self.relu_s5  = nn.PReLU()
        self.relu_s51 = nn.PReLU()

        self._initialize_weights(logger)
        self.fuse = nn.Conv2d(11, 1, 1, stride=1)
        
    def forward(self, x):
        all_fuse = list()
        features = self.features(x)
        sum1 = self.conv1_1_down(self.msblock1_1(features[0])) + \
            self.conv1_2_down(self.msblock1_2(features[1]))      # torch.Size([1, 64, 256, 256])
        sum1= self.relu_sum1(sum1)
        s1  = self.relu_s1(self.score_dsn1(sum1))
        s11 = self.relu_s11(self.score_dsn1_1(sum1))

        sum2 = self.conv2_1_down(self.msblock2_1(features[2])) + \
            self.conv2_2_down(self.msblock2_2(features[3]))          # torch.Size([1, 128, 128, 128])
        sum2 = self.relu_sum2(sum2)
        s2  = self.relu_s2(self.score_dsn2(sum2))
        s21 = self.relu_s21(self.score_dsn2_1(sum2))
        s2  = self.upsample_2(s2)
        s21 = self.upsample_2(s21)
        s2  = crop(s2, x, 1, 1)
        s21 = crop(s21, x, 1, 1)

        sum3 = self.conv3_1_down(self.msblock3_1(features[4])) + \
            self.conv3_2_down(self.msblock3_2(features[5])) + \
            self.conv3_3_down(self.msblock3_3(features[6]))          # torch.Size([1, 256, 64, 64])
        sum3 = self.relu_sum3(sum3)
        s3  = self.relu_s3(self.score_dsn3(sum3))
        s31 = self.relu_s31(self.score_dsn3_1(sum3))
        s3  = self.upsample_4(s3)
        s31 = self.upsample_4(s31)               
        s3  = crop(s3, x, 2, 2)
        s31 = crop(s31, x, 2, 2)

        sum4 = self.conv4_1_down(self.msblock4_1(features[7])) + \
            self.conv4_2_down(self.msblock4_2(features[8])) + \
            self.conv4_3_down(self.msblock4_3(features[9]))
        sum4 = self.relu_sum4(sum4)
        s4  = self.relu_s4(self.score_dsn4(sum4))
        s41 = self.relu_s41(self.score_dsn4_1(sum4))
        s4  = self.upsample_8(s4)
        s41 = self.upsample_8(s41)                                    # torch.Size([1, 1, 264, 264])
        s4  = crop(s4, x, 4, 4)
        s41 = crop(s41, x, 4, 4)

        sum5 = self.conv5_1_down(self.msblock5_1(features[10])) + \
            self.conv5_2_down(self.msblock5_2(features[11])) + \
            self.conv5_3_down(self.msblock5_3(features[12]))          # torch.Size([1, 512, 31, 31])features[12]
        sum5 = self.relu_sum5(sum5)
        s5  = self.relu_s5(self.score_dsn5(sum5))
        s51 = self.relu_s51(self.score_dsn5_1(sum5))
        s5  = self.upsample_8_5(s5)
        s51 = self.upsample_8_5(s51)
        s5  = crop(s5, x, 0, 0)
        s51 = crop(s51, x, 0, 0)

        pam_feat   = self.pam_relu1(self.pam_conv1(features[12]))
        pam_feat   = self.pam(pam_feat)
        pam_conv   = self.pam_relu2(self.pam_conv2(pam_feat))
        pam_conv   = self.pam_drop(pam_conv) 
        pam_output = self.pam_relu3(self.pam_conv3(pam_conv))
        pam_output = self.upsample_8_5_pam(pam_output)                 # torch.Size([1, 1, 256, 256])
   
        nam_feat   = self.nam_relu1(self.nam_conv1(features[12]))
        nam_feat   = self.nam(nam_feat)
        nam_conv   = self.nam_relu2(self.nam_conv2(nam_feat))
        nam_conv   = self.nam_drop(nam_conv) 
        nam_output = self.nam_relu3(self.nam_conv3(nam_conv))
        nam_output = self.upsample_8_5_nam(nam_output) 
        
        anttion_sum = pam_conv + nam_conv
        pana_output = self.pana_drop(anttion_sum)
        pana_output = self.pana_conv1(pana_output)
        pana_output = self.upsample_8_5_pana(pana_output)
      
        o1, o2, o3, o4, o5 = s1.detach(), s2.detach(), s3.detach(), s4.detach(), s5.detach()
        o11, o21, o31, o41, o51 = s11.detach(), s21.detach(), s31.detach(), s41.detach(), s51.detach()
        
        p1_1_1 = s1  + pana_output
        p2_1_1 = s2  + o1  + pana_output  
        p3_1_1 = s3  + o2  + o1  + pana_output
        p4_1_1 = s4  + o3  + o2  + o1  + pana_output
        p5_1_1 = s5  + o4  + o3  + o2  + o1  + pana_output
        p1_2_2 = s11 + o21 + o31 + o41 + o51 + pana_output
        p2_2_2 = s21 + o31 + o41 + o51 + pana_output
        p3_2_2 = s31 + o41 + o51 + pana_output
        p4_2_2 = s41 + o51 + pana_output
        p5_2_2 = s51 + pana_output
      
        fuse = self.fuse(torch.cat([p1_1_1, p2_1_1, p3_1_1, p4_1_1, p5_1_1, p1_2_2, p2_2_2, p3_2_2, p4_2_2, p5_2_2, pana_output], 1))

        p1_1_1       = torch.sigmoid(p1_1_1)
        p2_1_1       = torch.sigmoid(p2_1_1)
        p3_1_1       = torch.sigmoid(p3_1_1)
        p4_1_1       = torch.sigmoid(p4_1_1)
        p5_1_1       = torch.sigmoid(p5_1_1)
        p1_2_2       = torch.sigmoid(p1_2_2)
        p2_2_2       = torch.sigmoid(p2_2_2)
        p3_2_2       = torch.sigmoid(p3_2_2)
        p4_2_2       = torch.sigmoid(p4_2_2)
        p5_2_2       = torch.sigmoid(p5_2_2)                       
        pam_output   = torch.sigmoid(pam_output)
        nam_output   = torch.sigmoid(nam_output)
        pana_output  = torch.sigmoid(pana_output)
        fuse         = torch.sigmoid(fuse)
        
        all_fuse.append(fuse)
        all_fuse.append(p1_1_1)
        all_fuse.append(p2_1_1)
        all_fuse.append(p3_1_1)
        all_fuse.append(p4_1_1)
        all_fuse.append(p5_1_1)
        all_fuse.append(p1_2_2)
        all_fuse.append(p2_2_2)
        all_fuse.append(p3_2_2)
        all_fuse.append(p4_2_2)
        all_fuse.append(p5_2_2)
        all_fuse.append(pam_output)
        all_fuse.append(nam_output)
        all_fuse.append(pana_output)
        
        return all_fuse

    
    def _initialize_weights(self, logger=None):
        for name, param in self.state_dict().items():
            if self.pretrain and 'features' in name:
                continue
            # elif 'down' in name:
            #     param.zero_()
            elif 'upsample' in name:
                if logger:
                    logger.info('init upsamle layer %s ' % name)
                k = int(name.split('.')[0].split('_')[1])
                param.copy_(get_upsampling_weight(1, 1, k*2))
            elif 'fuse' in name:
                if logger:
                    logger.info('init params %s ' % name)
                if 'bias' in name:
                    param.zero_()
                else:
                    nn.init.constant_(param, 0.080)
            else:
                if logger:
                    logger.info('init params %s ' % name)
                if 'bias' in name:
                    param.zero_()
                else:
                    param.normal_(0, 0.01)


def build_model():

    return CANet('/home/panzefeng/All_code/BDCN_salient_detection/weights/vgg16.pth')

if __name__ == '__main__':
   
    net = build_model()
    img = torch.rand((1, 3, 256, 256))  # output is (1,256,256)!
    net = net.to(torch.device('cuda:0'))
    img = img.to(torch.device('cuda:0'))
    out = net(img)
    for x in net(img):
        print (x.data.shape)    