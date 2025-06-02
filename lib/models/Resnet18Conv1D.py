import torch
import torch.nn as nn

class Resnet18Conv1D(nn.Module):

    def __init__(self, output_classes, input_channels=4, kernel_size=3):
        super(Resnet18Conv1D, self).__init__()

        self.kernel_size = kernel_size
        self.padding     = kernel_size // 2

        self.residual_1 = self.__residual_block(input_channels, 64)
        self.residual_2 = self.__residual_block(64, 128)
        self.residual_3 = self.__residual_block(128, 128)
        self.residual_4 = self.__residual_block(128, 256)
        self.residual_5 = self.__residual_block(256, 512)

        self.identity_expand1 = nn.Conv1d(input_channels, 64, kernel_size=1)
        self.identity_expand2 = nn.Conv1d(64, 128, kernel_size=1)
        self.identity_expand3 = nn.Conv1d(128, 256, kernel_size=1)
        self.identity_expand4 = nn.Conv1d(256, 512, kernel_size=1)

        self.activation = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, output_classes)
        self.dropout = nn.Dropout(0.2)

    def __residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding),
            nn.BatchNorm1d(out_channels)
        )

    def forward_block(self, x, block, identity_expand):

        identity = x

        if x.size(1) != identity_expand.out_channels:
            identity = identity_expand(x)

        out = block(x)
        out += identity
        out = self.activation(out)
        return out


    def main_network(self, x):
        x = self.forward_block(x, self.residual_1, self.identity_expand1)
        x = self.forward_block(x, self.residual_2, self.identity_expand2)
        x = self.forward_block(x, self.residual_3, self.identity_expand2)
        x = self.forward_block(x, self.residual_4, self.identity_expand3)
        x = self.forward_block(x, self.residual_5, self.identity_expand4)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def forward(self, x):
        x = self.main_network(x)
        x = torch.softmax(x, dim=1)
        return x

    def get_embedding(self, x):
        x = self.main_network(x)
        return x

    def perform_embedding_pca(self, x):
        x = self.get_embedding(x)
        x = x.detach().numpy()

        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        x   = pca.fit_transform(x)
        return x