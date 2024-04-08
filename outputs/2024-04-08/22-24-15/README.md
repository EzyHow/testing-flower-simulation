# This directory contains log and results generated after changing the model to resnet18 as given below ( added torchsummary):

```

class Net(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()

        self.model = models.resnet18()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        summary(self.model, input_size=(1, 28, 28))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x
```

# Check main.log, same simulation setup ran without any error.