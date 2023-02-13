import torch.nn as nn
import torch.nn.functional as F
import torch

class MLPnet(torch.nn.Module):
    def __init__(self, node_number, batch_size, k_hop):
        super(MLPnet,self).__init__()
        self.node_number = node_number
        self.batch_size = batch_size
        self.k_hop = k_hop
        self.aggregate_weightT = torch.nn.Parameter(torch.rand(1, 1, node_number))
        
        self.features = nn.Sequential(
            torch.nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),  # 12
            nn.ReLU(inplace=True),
            torch.nn.Linear(256, 2),
        )

    def forward(self, x_T):
        x_T = torch.matmul(self.aggregate_weightT, x_T)
        x = self.features(x_T.view(x_T.size(0),-1))

        return F.log_softmax(x, dim=1), x

class GNNnet(torch.nn.Module):
      def __init__(self, node_number, batch_size, k_hop):
          super(GNNnet,self).__init__()
          self.node_number = node_number
          self.batch_size = batch_size
          self.k_hop = k_hop
          self.aggregate_weightT = torch.nn.Parameter(torch.ones(1, 1, node_number))
          
          self.features = nn.Sequential(
              torch.nn.Linear(256, 256),
              nn.ReLU(inplace=True),
              torch.nn.Linear(256, 256),
              nn.ReLU(inplace=True),
              torch.nn.Linear(256, 256),
              nn.ReLU(inplace=True),
              torch.nn.Linear(256, 256),  # 12
              nn.ReLU(inplace=True),
              torch.nn.Linear(256, 2),
          )

      def forward(self, x_T):
      
          tmp_x_T = x_T
          for _ in range(self.k_hop):
              tmp_x_T = torch.matmul(tmp_x_T, x_T)
          x_T = torch.matmul(self.aggregate_weightT, tmp_x_T)
          x = self.features(x_T.view(x_T.size(0),-1))

          return F.log_softmax(x, dim=1), x

class CNN1Dnet(torch.nn.Module):
      def __init__(self, node_number, batch_size, k_hop):
          super(CNN1Dnet,self).__init__()
          self.node_number = node_number
          self.batch_size = batch_size
          self.aggregate_weight = torch.ones(1, 1, node_number)
          self.k_hop = k_hop

          self.features = nn.Sequential(
              nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
              nn.ReLU(inplace=True),
              nn.MaxPool1d(kernel_size=2),
              nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
              nn.ReLU(inplace=True),
              nn.MaxPool1d(kernel_size=2),
              nn.Conv1d(16, 32, kernel_size=9, stride=1, padding=4),
              nn.ReLU(inplace=True),
              nn.MaxPool1d(kernel_size=2),
              nn.Conv1d(32, 32, kernel_size=9, stride=1, padding=4),
              nn.ReLU(inplace=True),
              nn.MaxPool1d(kernel_size=2),
          )

          self.classifier = nn.Sequential(
              nn.Linear(16 * 32, 64),
              nn.ReLU(inplace=True),
              nn.Linear(64, 5),
          )

      def forward(self, x_T):
      
          x_T = torch.matmul(self.aggregate_weight, x_T)
          x_T = self.features(x_T)
          x = self.classifier(x_T.view(x_T.size(0),-1))

          return F.log_softmax(x, dim=1), x

class CNN2Dnet(torch.nn.Module):
      def __init__(self, node_number, batch_size, k_hop):
          super(CNN2Dnet,self).__init__()
          self.node_number = node_number
          self.batch_size = batch_size
          self.k_hop = k_hop

          self.features = nn.Sequential(
              nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
              nn.BatchNorm2d(8),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2),
              nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
              nn.BatchNorm2d(16),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2),
              nn.Conv2d(16, 32, kernel_size=9, stride=1, padding=4),
              nn.BatchNorm2d(32),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2),
              nn.Conv2d(32, 32, kernel_size=9, stride=1, padding=4),
              nn.BatchNorm2d(32),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2),
          )

          self.classifier = nn.Sequential(
              nn.Linear(16 * 16 * 32, 64),
              nn.ReLU(inplace=True),
              nn.Linear(64, 2),
          )

      def forward(self, x_T):

          x_T = self.features(x_T)
          x = self.classifier(x_T.view(x_T.size(0), -1))

          return F.log_softmax(x, dim=1), x

class SSGCNnet(torch.nn.Module):
      def __init__(self, node_number, batch_size, k_hop):
          super(SSGCNnet,self).__init__()
          self.node_number = node_number
          self.batch_size = batch_size
          self.k_hop = k_hop
          self.aggregate_weightT = torch.nn.Parameter(torch.ones(1, 1, node_number))
          
          self.features = nn.Sequential(
              nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
              nn.BatchNorm1d(8),
              nn.ReLU(inplace=True),
              nn.MaxPool1d(kernel_size=2),
              nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
              nn.BatchNorm1d(16),
              nn.ReLU(inplace=True),
              nn.MaxPool1d(kernel_size=2),
              nn.Conv1d(16, 32, kernel_size=9, stride=1, padding=4),
              nn.BatchNorm1d(32),
              nn.ReLU(inplace=True),
              nn.MaxPool1d(kernel_size=2),
              nn.Conv1d(32, 32, kernel_size=9, stride=1, padding=4),
              nn.BatchNorm1d(32),
              nn.ReLU(inplace=True),
              nn.MaxPool1d(kernel_size=2),
          )

          self.classifier = nn.Sequential(
              nn.Linear(16 * 32, 64),
              nn.BatchNorm1d(64),
              nn.ReLU(inplace=True),
              nn.Linear(64, 2),
          )

      def forward(self, x_T):
      
          x_T = torch.matmul(self.aggregate_weightT, x_T)

          x_T = self.features(x_T)
          x = self.classifier(x_T.view(x_T.size(0), -1))

          return F.log_softmax(x, dim=1), x
