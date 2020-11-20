
import torch
import torch.nn as nn


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        out=x*psi

        return out


class Att_Unet(nn.Module):
    def __init__(self,output_ch=1):
        super(Att_Unet,self).__init__()


        self.base_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                         in_channels=3,out_channels=1, init_features=32,
                                          pretrained=True,verbose=False)
        self.base_layers = list(self.base_model.children())
        self.layer0=nn.Sequential(*self.base_layers[0])
        self.layer1=nn.Sequential(*self.base_layers[1:3])
        self.layer2=nn.Sequential(*self.base_layers[3:5])
        self.layer3=nn.Sequential(*self.base_layers[5:7])

        self.layer4=nn.Sequential(*self.base_layers[7:9])



        self.Up5 = self.base_layers[9]
        self.Att5 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv5 = nn.Sequential(*self.base_layers[10])

        self.Up4 = self.base_layers[11]
        self.Att4 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv4 =nn.Sequential(*self.base_layers[12])

        self.Up3 = self.base_layers[13]
        self.Att3 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv3 = nn.Sequential(*self.base_layers[14])

        self.Up2 = self.base_layers[15]
        self.Att2 = Attention_block(F_g=32,F_l=32,F_int=16)
        self.Up_conv2 = nn.Sequential(*self.base_layers[16])


        self.Conv_1x1 =self.base_layers[17]


    def forward(self,x):
        # encoding path
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)

        x5 = self.layer4(x4)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
