
import torch
from torch import nn
from torch.nn.modules.upsampling import Upsample
import torch.nn.functional as F
import numpy as np


planes = [32,64, 128, 256, 512]
class Block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        #planes = [64, 128, 256, 512,1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = Block(input_channels, planes[0], planes[0])
        self.conv1_0 = Block(planes[0], planes[1], planes[1])
        self.conv2_0 = Block(planes[1], planes[2], planes[2])
        self.conv3_0 = Block(planes[2], planes[3], planes[3])
        self.conv4_0 = Block(planes[3], planes[4], planes[4])

        self.conv3_1 = Block(planes[3]+planes[4], planes[3], planes[3])
        self.conv2_2 = Block(planes[2]+planes[3], planes[2], planes[2])
        self.conv1_3 = Block(planes[1]+planes[2], planes[1], planes[1])
        self.conv0_4 = Block(planes[0]+planes[1], planes[0], planes[0])

        self.final = nn.Conv2d(planes[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        #planes = [32,64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2,ceil_mode=True)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = Block(input_channels, planes[0], planes[0])
        self.conv1_0 = Block(planes[0], planes[1], planes[1])
        self.conv2_0 = Block(planes[1], planes[2], planes[2])
        self.conv3_0 = Block(planes[2], planes[3], planes[3])
        self.conv4_0 = Block(planes[3], planes[4], planes[4])

        self.conv0_1 = Block(planes[0]+planes[1], planes[0], planes[0])
        self.conv1_1 = Block(planes[1]+planes[2], planes[1], planes[1])
        self.conv2_1 = Block(planes[2]+planes[3], planes[2], planes[2])
        self.conv3_1 = Block(planes[3]+planes[4], planes[3], planes[3])

        self.conv0_2 = Block(planes[0]*2+planes[1], planes[0], planes[0])
        self.conv1_2 = Block(planes[1]*2+planes[2], planes[1], planes[1])
        self.conv2_2 = Block(planes[2]*2+planes[3], planes[2], planes[2])

        self.conv0_3 = Block(planes[0]*3+planes[1], planes[0], planes[0])
        self.conv1_3 = Block(planes[1]*3+planes[2], planes[1], planes[1])

        self.conv0_4 = Block(planes[0]*4+planes[1], planes[0], planes[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(planes[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(planes[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(planes[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(planes[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(planes[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

        

class Down(nn.Module):
    def __init__(self,inplane,plane,stride,dropout=0.5):
        super().__init__()
        self.conv=nn.Conv2d(inplane,plane,1)
        self.bn=nn.BatchNorm2d(plane)
        self.relu=nn.ReLU(inplace=True)
        self.dropout=nn.Dropout(dropout)
        self.avgpool=nn.AvgPool2d(stride,stride)
        #self.block=Block()

    def forward(self,x):
        out=self.avgpool(x)
        out=self.conv(out)
        #out=self.bn(out)
        #out=self.relu(out)
        return out

class Up(nn.Module):
    def __init__(self,inplane,plane,scale_factor):
        super().__init__()
        self.ps=nn.PixelShuffle(scale_factor)
        self.conv=nn.Conv2d(inplane,plane,1) #3, padding=1
        self.bn=nn.BatchNorm2d(plane)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        out=self.ps(x)
        out=self.conv(out)
        #out=self.bn(out)
        #out=self.relu(out)
        return out


class WSUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        
        usplanes=np.array([2,4,8,16,32])*int(planes[0]/32)

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.encoder0 = Block(input_channels, planes[0], planes[0])
        self.ds0_2=Down(planes[0],planes[1],4)
        self.ds0_3=Down(planes[0],planes[2],8)
        self.ds0_4=Down(planes[0],planes[3],16)

        self.encoder1 = Block(planes[0], planes[1], planes[1])
        self.ds1_3=Down(planes[1],planes[2],4)
        self.ds1_4=Down(planes[1],planes[3],8)

        self.encoder2 = Block(planes[1], planes[2], planes[2])
        self.ds2_4=Down(planes[2],planes[3],4)

        self.encoder3 = Block(planes[2], planes[3], planes[3])

        self.encoder4 = Block(planes[3], planes[4], planes[4])
        self.us4_2=Up(usplanes[4],planes[2],4)
        self.us4_1=Up(usplanes[2],planes[1],8)
        self.us4_0=Up(usplanes[0],planes[0],16)

        self.decoder3 = Block(planes[4]+planes[3], planes[3], planes[3]) #planes[3]+planes[4]
        self.us3_1=Up(usplanes[3],planes[1],4)
        self.us3_0=Up(usplanes[1],planes[0],8)

        self.decoder2 = Block(planes[3]+planes[2], planes[2], planes[2]) #planes[2]+planes[3]+64
        self.us2_0=Up(usplanes[2],planes[0],4)

        self.decoder1 = Block(planes[2]+planes[1], planes[1], planes[1]) #planes[1]+planes[2]+16+32

        self.decoder0 = Block(planes[1]+planes[0], planes[0], planes[0])  #planes[0]+planes[1]+4+8+16

        self.final = nn.Conv2d(planes[0], num_classes, kernel_size=1)

        self.ew2 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.ew0_2 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

        self.ew3 = torch.nn.Parameter(torch.Tensor([0.33]), requires_grad=True)
        self.ew0_3 = torch.nn.Parameter(torch.Tensor([0.33]), requires_grad=True)
        self.ew1_3 = torch.nn.Parameter(torch.Tensor([0.33]), requires_grad=True)

        self.ew4 = torch.nn.Parameter(torch.Tensor([0.25]), requires_grad=True)
        self.ew0_4 = torch.nn.Parameter(torch.Tensor([0.25]), requires_grad=True)        
        self.ew1_4 = torch.nn.Parameter(torch.Tensor([0.25]), requires_grad=True)
        self.ew2_4 = torch.nn.Parameter(torch.Tensor([0.25]), requires_grad=True)

      
        self.dw2 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.dw4_2 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

        self.dw1 = torch.nn.Parameter(torch.Tensor([0.33]), requires_grad=True)
        self.dw3_1 = torch.nn.Parameter(torch.Tensor([0.33]), requires_grad=True)
        self.dw4_1 = torch.nn.Parameter(torch.Tensor([0.33]), requires_grad=True)

        self.dw0 = torch.nn.Parameter(torch.Tensor([0.25]), requires_grad=True)
        self.dw2_0 = torch.nn.Parameter(torch.Tensor([0.25]), requires_grad=True)
        self.dw3_0 = torch.nn.Parameter(torch.Tensor([0.25]), requires_grad=True)        
        self.dw4_0 = torch.nn.Parameter(torch.Tensor([0.25]), requires_grad=True)


    def forward(self, input):
        x0 = self.encoder0(input)
        x1 = self.encoder1(self.pool(x0))
   
        x2 = self.encoder2(self.ew2*self.pool(x1)+self.ew0_2*self.ds0_2(x0))
        x3 = self.encoder3(self.ew3*self.pool(x2)+self.ew0_3*self.ds0_3(x0)+self.ew1_3*self.ds1_3(x1))
        x4 = self.encoder4(self.ew4*self.pool(x3)+self.ew0_4*self.ds0_4(x0)+self.ew1_4*self.ds1_4(x1)+self.ew2_4*self.ds2_4(x2))

        y3 = self.decoder3(torch.cat([x3, self.up(x4)], 1))
        y2 = self.decoder2(torch.cat([self.dw2*x2+self.dw4_2*self.us4_2(x4), self.up(y3)], 1))
        y1 = self.decoder1(torch.cat([self.dw1*x1+self.dw4_1*self.us4_1(x4)+self.dw3_1*self.us3_1(y3), self.up(y2)], 1))
        y0 = self.decoder0(torch.cat([self.dw0*x0+self.dw4_0*self.us4_0(x4)+self.dw3_0*self.us3_0(y3)+self.dw2_0*self.us2_0(y2), self.up(y1)], 1))

        output = self.final(y0)
        return output



class WS1UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        
        usplanes=np.array([2,4,8,16,32])*int(planes[0]/32)

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.encoder0 = Block(input_channels, planes[0], planes[0])
        self.ds0_2=Down(planes[0],planes[1],4)
       

        self.encoder1 = Block(planes[0], planes[1], planes[1])
        self.ds1_3=Down(planes[1],planes[2],4)

        self.encoder2 = Block(planes[1], planes[2], planes[2])
        self.ds2_4=Down(planes[2],planes[3],4)

        self.encoder3 = Block(planes[2], planes[3], planes[3])

        self.encoder4 = Block(planes[3], planes[4], planes[4])
        self.us4_2=Up(usplanes[4],planes[2],4)

        self.decoder3 = Block(planes[4]+planes[3], planes[3], planes[3]) #planes[3]+planes[4]
        self.us3_1=Up(usplanes[3],planes[1],4)

        self.decoder2 = Block(planes[3]+planes[2], planes[2], planes[2]) #planes[2]+planes[3]+64
        self.us2_0=Up(usplanes[2],planes[0],4)

        self.decoder1 = Block(planes[2]+planes[1], planes[1], planes[1]) #planes[1]+planes[2]+16+32

        self.decoder0 = Block(planes[1]+planes[0], planes[0], planes[0])  #planes[0]+planes[1]+4+8+16

        self.final = nn.Conv2d(planes[0], num_classes, kernel_size=1)

        self.ew2 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.ew0_2 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

        self.ew3 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.ew1_3 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

        self.ew4 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.ew2_4 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

      
        self.dw2 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.dw4_2 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

        self.dw1 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.dw3_1 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

        self.dw0 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.dw2_0 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)


    def forward(self, input):
        x0 = self.encoder0(input)
        x1 = self.encoder1(self.pool(x0))
   
        x2 = self.encoder2(self.ew2*self.pool(x1)+self.ew0_2*self.ds0_2(x0))
        x3 = self.encoder3(self.ew3*self.pool(x2)+self.ew1_3*self.ds1_3(x1))
        x4 = self.encoder4(self.ew4*self.pool(x3)+self.ew2_4*self.ds2_4(x2))

        y3 = self.decoder3(torch.cat([x3, self.up(x4)], 1))
        y2 = self.decoder2(torch.cat([self.dw2*x2+self.dw4_2*self.us4_2(x4), self.up(y3)], 1))
        y1 = self.decoder1(torch.cat([self.dw1*x1+self.dw3_1*self.us3_1(y3), self.up(y2)], 1))
        y0 = self.decoder0(torch.cat([self.dw0*x0+self.dw2_0*self.us2_0(y2), self.up(y1)], 1))

        output = self.final(y0)
        return output

class WS2UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        
        usplanes=np.array([2,4,8,16,32])*int(planes[0]/32)

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.encoder0 = Block(input_channels, planes[0], planes[0])
        self.ds0_2=Down(planes[0],planes[1],4)
        self.ds0_3=Down(planes[0],planes[2],8)

        self.encoder1 = Block(planes[0], planes[1], planes[1])
        self.ds1_3=Down(planes[1],planes[2],4)
        self.ds1_4=Down(planes[1],planes[3],8)

        self.encoder2 = Block(planes[1], planes[2], planes[2])
        self.ds2_4=Down(planes[2],planes[3],4)

        self.encoder3 = Block(planes[2], planes[3], planes[3])

        self.encoder4 = Block(planes[3], planes[4], planes[4])
        self.us4_2=Up(usplanes[4],planes[2],4)
        self.us4_1=Up(usplanes[2],planes[1],8)

        self.decoder3 = Block(planes[4]+planes[3], planes[3], planes[3]) #planes[3]+planes[4]
        self.us3_1=Up(usplanes[3],planes[1],4)
        self.us3_0=Up(usplanes[1],planes[0],8)

        self.decoder2 = Block(planes[3]+planes[2], planes[2], planes[2]) #planes[2]+planes[3]+64
        self.us2_0=Up(usplanes[2],planes[0],4)

        self.decoder1 = Block(planes[2]+planes[1], planes[1], planes[1]) #planes[1]+planes[2]+16+32

        self.decoder0 = Block(planes[1]+planes[0], planes[0], planes[0])  #planes[0]+planes[1]+4+8+16

        self.final = nn.Conv2d(planes[0], num_classes, kernel_size=1)

        self.ew2 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.ew0_2 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

        self.ew3 = torch.nn.Parameter(torch.Tensor([0.33]), requires_grad=True)
        self.ew0_3 = torch.nn.Parameter(torch.Tensor([0.33]), requires_grad=True)
        self.ew1_3 = torch.nn.Parameter(torch.Tensor([0.33]), requires_grad=True)

        self.ew4 = torch.nn.Parameter(torch.Tensor([0.33]), requires_grad=True)        
        self.ew1_4 = torch.nn.Parameter(torch.Tensor([0.33]), requires_grad=True)
        self.ew2_4 = torch.nn.Parameter(torch.Tensor([0.33]), requires_grad=True)

      
        self.dw2 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.dw4_2 = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

        self.dw1 = torch.nn.Parameter(torch.Tensor([0.33]), requires_grad=True)
        self.dw3_1 = torch.nn.Parameter(torch.Tensor([0.33]), requires_grad=True)
        self.dw4_1 = torch.nn.Parameter(torch.Tensor([0.33]), requires_grad=True)

        self.dw0 = torch.nn.Parameter(torch.Tensor([0.33]), requires_grad=True)
        self.dw2_0 = torch.nn.Parameter(torch.Tensor([0.33]), requires_grad=True)
        self.dw3_0 = torch.nn.Parameter(torch.Tensor([0.33]), requires_grad=True)


    def forward(self, input):
        x0 = self.encoder0(input)
        x1 = self.encoder1(self.pool(x0))
   
        x2 = self.encoder2(self.ew2*self.pool(x1)+self.ew0_2*self.ds0_2(x0))
        x3 = self.encoder3(self.ew3*self.pool(x2)+self.ew0_3*self.ds0_3(x0)+self.ew1_3*self.ds1_3(x1))
        x4 = self.encoder4(self.ew4*self.pool(x3)+self.ew1_4*self.ds1_4(x1)+self.ew2_4*self.ds2_4(x2))

        y3 = self.decoder3(torch.cat([x3, self.up(x4)], 1))
        y2 = self.decoder2(torch.cat([self.dw2*x2+self.dw4_2*self.us4_2(x4), self.up(y3)], 1))
        y1 = self.decoder1(torch.cat([self.dw1*x1+self.dw4_1*self.us4_1(x4)+self.dw3_1*self.us3_1(y3), self.up(y2)], 1))
        y0 = self.decoder0(torch.cat([self.dw0*x0+self.dw3_0*self.us3_0(y3)+self.dw2_0*self.us2_0(y2), self.up(y1)], 1))

        output = self.final(y0)
        return output


class AdaptiveUNetL3(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        planes = [32,64, 128, 256]
        usplanes=np.array([2,4,8,16,32])*int(planes[0]/32)

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.encoder0 = Block(input_channels, planes[0], planes[0])
        self.ds0_2=Down(planes[0],planes[1],4)
        self.ds0_3=Down(planes[0],planes[2],8)
        self.ds0_4=Down(planes[0],planes[3],16)

        self.encoder1 = Block(planes[0], planes[1], planes[1])
        self.ds1_3=Down(planes[1],planes[2],4)
        self.ds1_4=Down(planes[1],planes[3],8)

        self.encoder2 = Block(planes[1], planes[2], planes[2])
        self.ds2_4=Down(planes[2],planes[3],4)

        self.encoder3 = Block(planes[2], planes[3], planes[3])


        self.us3_1=Up(usplanes[3],planes[1],4)
        self.us3_0=Up(usplanes[1],planes[0],8)

        self.decoder2 = Block(planes[3]+planes[2], planes[2], planes[2]) #planes[2]+planes[3]+64
        self.us2_0=Up(usplanes[2],planes[0],4)

        self.decoder1 = Block(planes[2]+planes[1], planes[1], planes[1]) #planes[1]+planes[2]+16+32

        self.decoder0 = Block(planes[1]+planes[0], planes[0], planes[0])  #planes[0]+planes[1]+4+8+16

        self.final = nn.Conv2d(planes[0], num_classes, kernel_size=1)

        self.ew2 = torch.nn.Parameter(torch.Tensor([1/2]), requires_grad=True)
        self.ew0_2 = torch.nn.Parameter(torch.Tensor([1/2]), requires_grad=True)

        self.ew3 = torch.nn.Parameter(torch.Tensor([1/3]), requires_grad=True)
        self.ew0_3 = torch.nn.Parameter(torch.Tensor([1/3]), requires_grad=True)
        self.ew1_3 = torch.nn.Parameter(torch.Tensor([1/3]), requires_grad=True)

        
        self.dw1 = torch.nn.Parameter(torch.Tensor([1/3]), requires_grad=True)
        self.dw3_1 = torch.nn.Parameter(torch.Tensor([1/3]), requires_grad=True)

        self.dw0 = torch.nn.Parameter(torch.Tensor([1/4]), requires_grad=True)
        self.dw2_0 = torch.nn.Parameter(torch.Tensor([1/4]), requires_grad=True)
        self.dw3_0 = torch.nn.Parameter(torch.Tensor([1/4]), requires_grad=True)


    def forward(self, input):
        x0 = self.encoder0(input)
        x1 = self.encoder1(self.pool(x0))
   
        x2 = self.encoder2(self.ew2*self.pool(x1)+self.ew0_2*self.ds0_2(x0))
        x3 = self.encoder3(self.ew3*self.pool(x2)+self.ew0_3*self.ds0_3(x0)+self.ew1_3*self.ds1_3(x1))
    
     
        y2 = self.decoder2(torch.cat([x2, self.up(x3)], 1))
        y1 = self.decoder1(torch.cat([self.dw1*x1+self.dw3_1*self.us3_1(x3), self.up(y2)], 1))
        y0 = self.decoder0(torch.cat([self.dw0*x0+self.dw3_0*self.us3_0(x3)+self.dw2_0*self.us2_0(y2), self.up(y1)], 1))

        output = self.final(y0)
        return output

class AdaptiveUNetL5(nn.Module):
	def __init__(self, num_classes, input_channels=3, **kwargs):
		super().__init__()

		planes = [32,64, 128, 256,512,1024]
		usplanes=np.array([1,2,4,8,16,32,64])*int(planes[0]/32)

		self.pool = nn.MaxPool2d(2, 2)
		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

		self.encoder0 = Block(input_channels, planes[0], planes[0])
		self.ds0_2=Down(planes[0],planes[1],4)
		self.ds0_3=Down(planes[0],planes[2],8)
		self.ds0_4=Down(planes[0],planes[3],16)
		self.ds0_5=Down(planes[0],planes[4],32)

		self.encoder1 = Block(planes[0], planes[1], planes[1])
		self.ds1_3=Down(planes[1],planes[2],4)
		self.ds1_4=Down(planes[1],planes[3],8)
		self.ds1_5=Down(planes[1],planes[4],16)

		self.encoder2 = Block(planes[1], planes[2], planes[2])
		self.ds2_4=Down(planes[2],planes[3],4)
		self.ds2_5=Down(planes[2],planes[4],8)

		self.encoder3 = Block(planes[2], planes[3], planes[3])
		self.ds3_5=Down(planes[3],planes[4],4)

		self.encoder4 = Block(planes[3], planes[4], planes[4])

		self.encoder5 = Block(planes[4], planes[5], planes[5])
		self.us5_3=Up(usplanes[6],planes[3],4)
		self.us5_2=Up(usplanes[4],planes[2],8)
		self.us5_1=Up(usplanes[2],planes[1],16)
		self.us5_0=Up(usplanes[0],planes[0],32)

		self.decoder4 = Block(planes[5]+planes[4], planes[4], planes[4])
		self.us4_2=Up(usplanes[5],planes[2],4)
		self.us4_1=Up(usplanes[3],planes[1],8)
		self.us4_0=Up(usplanes[1],planes[0],16)

		self.decoder3 = Block(planes[4]+planes[3], planes[3], planes[3]) #planes[3]+planes[4]
		self.us3_1=Up(usplanes[4],planes[1],4)
		self.us3_0=Up(usplanes[2],planes[0],8)

		self.decoder2 = Block(planes[3]+planes[2], planes[2], planes[2]) #planes[2]+planes[3]+64
		self.us2_0=Up(usplanes[3],planes[0],4)

		self.decoder1 = Block(planes[2]+planes[1], planes[1], planes[1]) #planes[1]+planes[2]+16+32

		self.decoder0 = Block(planes[1]+planes[0], planes[0], planes[0])  #planes[0]+planes[1]+4+8+16

		self.final = nn.Conv2d(planes[0], num_classes, kernel_size=1)

		self.ew2 = torch.nn.Parameter(torch.Tensor([1/2]), requires_grad=True)
		self.ew0_2 = torch.nn.Parameter(torch.Tensor([1/2]), requires_grad=True)

		self.ew3 = torch.nn.Parameter(torch.Tensor([1/3]), requires_grad=True)
		self.ew0_3 = torch.nn.Parameter(torch.Tensor([1/3]), requires_grad=True)
		self.ew1_3 = torch.nn.Parameter(torch.Tensor([1/3]), requires_grad=True)

		self.ew4 = torch.nn.Parameter(torch.Tensor([1/4]), requires_grad=True)
		self.ew0_4 = torch.nn.Parameter(torch.Tensor([1/4]), requires_grad=True)        
		self.ew1_4 = torch.nn.Parameter(torch.Tensor([1/4]), requires_grad=True)
		self.ew2_4 = torch.nn.Parameter(torch.Tensor([1/4]), requires_grad=True)

		self.ew4 = torch.nn.Parameter(torch.Tensor([1/4]), requires_grad=True)
		self.ew0_4 = torch.nn.Parameter(torch.Tensor([1/4]), requires_grad=True)        
		self.ew1_4 = torch.nn.Parameter(torch.Tensor([1/4]), requires_grad=True)
		self.ew2_4 = torch.nn.Parameter(torch.Tensor([1/4]), requires_grad=True)

		self.ew5 = torch.nn.Parameter(torch.Tensor([1/5]), requires_grad=True)
		self.ew0_5 = torch.nn.Parameter(torch.Tensor([1/5]), requires_grad=True)        
		self.ew1_5 = torch.nn.Parameter(torch.Tensor([1/5]), requires_grad=True)
		self.ew2_5 = torch.nn.Parameter(torch.Tensor([1/5]), requires_grad=True)
		self.ew3_5 = torch.nn.Parameter(torch.Tensor([1/5]), requires_grad=True)

		
		self.dw3 = torch.nn.Parameter(torch.Tensor([1/2]), requires_grad=True)
		self.dw5_3 = torch.nn.Parameter(torch.Tensor([1/2]), requires_grad=True)

		self.dw2 = torch.nn.Parameter(torch.Tensor([1/3]), requires_grad=True)
		self.dw4_2 = torch.nn.Parameter(torch.Tensor([1/3]), requires_grad=True)
		self.dw5_2 = torch.nn.Parameter(torch.Tensor([1/3]), requires_grad=True)

		self.dw1 = torch.nn.Parameter(torch.Tensor([1/4]), requires_grad=True)
		self.dw3_1 = torch.nn.Parameter(torch.Tensor([1/4]), requires_grad=True)
		self.dw4_1 = torch.nn.Parameter(torch.Tensor([1/4]), requires_grad=True)        
		self.dw5_1 = torch.nn.Parameter(torch.Tensor([1/4]), requires_grad=True)

		self.dw0 = torch.nn.Parameter(torch.Tensor([1/5]), requires_grad=True)
		self.dw2_0 = torch.nn.Parameter(torch.Tensor([1/5]), requires_grad=True)
		self.dw3_0 = torch.nn.Parameter(torch.Tensor([1/5]), requires_grad=True)        
		self.dw4_0 = torch.nn.Parameter(torch.Tensor([1/5]), requires_grad=True)
		self.dw5_0 = torch.nn.Parameter(torch.Tensor([1/5]), requires_grad=True)

	def forward(self, input):
		x0 = self.encoder0(input)
		x1 = self.encoder1(self.pool(x0))

		x2 = self.encoder2(self.ew2*self.pool(x1)+self.ew0_2*self.ds0_2(x0))
		x3 = self.encoder3(self.ew3*self.pool(x2)+self.ew0_3*self.ds0_3(x0)+self.ew1_3*self.ds1_3(x1))
		x4 = self.encoder4(self.ew4*self.pool(x3)+self.ew0_4*self.ds0_4(x0)+self.ew1_4*self.ds1_4(x1)+self.ew2_4*self.ds2_4(x2))
		x5 = self.encoder5(self.ew5*self.pool(x4)+self.ew0_5*self.ds0_5(x0)+self.ew1_5*self.ds1_5(x1)+self.ew2_5*self.ds2_5(x2)+self.ew3_5*self.ds3_5(x3))

		y4 = self.decoder4(torch.cat([x4, self.up(x5)], 1))
		y3 = self.decoder3(torch.cat([self.dw3*x3+self.dw5_3*self.us5_3(x5), self.up(y4)], 1))
		y2 = self.decoder2(torch.cat([self.dw2*x2+self.dw5_2*self.us5_2(x5)+self.dw4_2*self.us4_2(y4), self.up(y3)], 1))
		y1 = self.decoder1(torch.cat([self.dw1*x1+self.dw5_1*self.us5_1(x5)+self.dw4_1*self.us4_1(y4)+self.dw3_1*self.us3_1(y3), self.up(y2)], 1))
		y0 = self.decoder0(torch.cat([self.dw0*x0+self.dw5_0*self.us5_0(x5)+self.dw4_0*self.us4_0(y4)+self.dw3_0*self.us3_0(y3)+self.dw2_0*self.us2_0(y2), self.up(y1)], 1))

		output = self.final(y0)
		return output
