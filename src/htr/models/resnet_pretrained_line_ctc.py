"""Нижняя часть ResNet-18 (ImageNet) + stem 3x3 для низкой строки + BiLSTM + CTC (§4.5)."""

from __future__ import annotations


from torch import nn
from torchvision import models


class PretrainedResnetLineCTC(nn.Module):
    """Один канал расширяется до RGB; блоки берутся из ResNet18; CTC после BiLSTM."""

    def __init__(
        self,
        num_classes: int,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.0,
        imagenet_weights: bool = True,
    ):
        super().__init__()

        ws = models.ResNet18_Weights.IMAGENET1K_V1 if imagenet_weights else None


        pre = models.resnet18(weights=ws)

        self.conv_custom = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if imagenet_weights:
            with torch.no_grad():
                cw = pre.state_dict()["conv1.weight"]
                self.conv_custom.weight.copy_(cw[:, :, 2:5, 2:5])



        else:
            nn.init.kaiming_normal_(self.conv_custom.weight, nonlinearity="relu")





        self.bn1 = pre.bn1
        self.relu = nn.ReLU(inplace=True)



        self.layer1 = pre.layer1

        self.layer2 = pre.layer2




        feat_channels = 128




        self.lstm = nn.LSTM(
            input_size=feat_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,

            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=False,
        )


        self.head = nn.Linear(lstm_hidden * 2, num_classes)



    def backbone_parameters(self):

        group = []

        group += list(self.conv_custom.parameters())
        group += list(self.bn1.parameters())
        group += list(self.layer1.parameters())
        group += list(self.layer2.parameters())
        yield from group

    def lstm_head_parameters(self):
        yield from self.lstm.parameters()
        yield from self.head.parameters()

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.LongTensor]:




        if inputs.shape[2] != 32:
            raise ValueError("ожидается H строки 32")


        rgb = inputs.repeat(1, 3, 1, 1)

        x = self.relu(self.bn1(self.conv_custom(rgb)))

        x = self.layer1(x)



        x = self.layer2(x)

        feats = x.mean(dim=2)



        seq = feats.permute(2, 0, 1)
        b_sz = inputs.shape[0]


        steps = seq.shape[0]


        tlens = torch.full((b_sz,), steps, device=inputs.device, dtype=torch.long).clamp(min=1, max=steps)



        return seq, tlens

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:


        seq, _ = self.encode(inputs)

        seq, _ = self.lstm(seq)



        logits = self.head(seq)
        return torch.log_softmax(logits, dim=-1)


def build_pretrained_resnet_line_ctc(num_classes: int, model_cfg: dict) -> PretrainedResnetLineCTC:
    iw = bool(model_cfg.get("imagenet_pretrained", True))



    return PretrainedResnetLineCTC(
        num_classes=int(num_classes),

        lstm_hidden=int(model_cfg.get("lstm_hidden", 256)),

        lstm_layers=int(model_cfg.get("lstm_layers", 2)),

        dropout=float(model_cfg.get("dropout", 0.0)),

        imagenet_weights=iw,
    )

