import torch
import torch.nn as nn

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm1 = LRN(local_size=5, alpha=0.0001, beta=0.75)
  
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.norm2 = LRN(local_size=5, alpha=0.0001, beta=0.75)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
    
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, input):
        conv1 = self.conv1(input)
        relu1 = self.relu1(conv1)
        pool1 = self.pool1(relu1)
        norm1 = self.norm1(pool1)

        conv2 = self.conv2(norm1)
        relu2 = self.relu2(conv2)
        pool2 = self.pool2(relu2)
        norm2 = self.norm2(pool2)

        conv3 = self.conv3(norm2)
        relu3 = self.relu3(conv3)

        conv4 = self.conv4(relu3)
        relu4 = self.relu4(conv4)

        conv5 = self.conv5(relu4)
        relu5 = self.relu5(conv5)
        pool5 = self.pool5(relu5)
        return pool5

class Classifier(nn.Module):
    def __init__(self, num_classes=31):
        super(Classifier, self).__init__()

        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout()

        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout()
   
        self.fc8 = nn.Linear(4096, num_classes)

    def forward(self, input):
        input = input.view(input.size(0), 256 * 6 * 6)
        fc6 = self.fc6(input)
        relu6 = self.relu6(fc6)
        drop6 = self.drop6(relu6)

        fc7 = self.fc7(drop6)
        relu7 = self.relu7(fc7)
        drop7 = self.drop7(relu7)

        # fc8 = self.fc8(drop7)
        return drop7

def alexnet(pretrained=False):
  model = Extractor().cuda()
  print("backbone:alexnet")
#   if pretrained:
#     model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'))
  if pretrained == True:
    model_path = '/SSD/xzf/msda/prototype/alexnet_pretrain.py/bvlc_extractor.pth'
    print("Loading pretrained weights from %s" %(model_path))
    # model.load(model_path)
    model.load_state_dict(torch.load(model_path))
    # model.fc = torch.nn.Linear(2048, 1024)
  return model
def C(pretrained=False):
  model = Classifier(num_classes=31).cuda()
  print("backbone:alexnet")
#   if pretrained:
#     model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'))
  if pretrained == True:
    model_path = '/SSD/xzf/msda/prototype/alexnet_pretrain.py/bvlc_s1_cls.pth'
    print("Loading pretrained weights from %s" %(model_path))
    # model.load(model_path)
    model.load_state_dict(torch.load(model_path))
    # model.fc = torch.nn.Linear(2048, 1024)
  return model

class alex(nn.Module):

    def __init__(self, num_classes=31):
        super(alex, self).__init__()
        self.sharedNet = alexnet(True)
        self.sonnet1 = C(True)
       
    def forward(self, x):
        x = self.sharedNet(x)
        x = self.sonnet1(x)
        return x
    
# x = torch.randn([2,3,252,252]).cuda()  
# A = alex()
# y = A(x)  