import torch.nn as nn

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        pass
       # x = F.relu(self.conv1(x))
       # return F.relu(self.conv2(x))



if __name__ == "__main__":
    print("hi")
    m = Model2().cuda()
    print(m.parameters)
    print("by")


