import torch.nn as nn
import torch

class Embedder(nn.Module):

  def __init__(self,Nf,Nt,Nh,K):
    
    """
    Input to the system is a tensor with shape (B,F,T)

    1) LSTM will take input with shape (B,T,F) with Bidirectional & 4 layer system and output
      with shape (B,T,2*feat_size) - 2 for bidirectional

    2) The fully connected layer will take the input (B,T,2*feat_size) and return output with
    (B,T,F*K) 

    3) Reshape into the following (B,TF,K)
    """

    super(Embedder,self).__init__()

    self.Nf = Nf
    self.Nt = Nt
    self.K = K
    self.num_layers = 4
    self.D = 2 # For bidirectional
    self.Nh = Nh

    self.lstm = nn.LSTM(input_size = Nf,
                        hidden_size = Nh, 
                        num_layers = 4, 
                        bidirectional = True,
                        batch_first = True
    )

    self.FC = nn.Linear(in_features=2*Nh,
                        out_features=Nf*K
    )
  
  def forward(self,X):

    # First, we need to define the hidden inputs
    Nb = X.shape[0]
    h0 = torch.rand(self.num_layers*self.D,Nb,self.Nh)
    c0 = torch.rand(self.num_layers*self.D,Nb,self.Nh)

    # Swap axis for time and frequency
    X = torch.swapaxes(X,-2,-1)

    # Input to the LSTM
    Y,_ = self.lstm(X,(h0,c0))

    # Input to FC
    embedding = self.FC(Y)

    # Reshape to the format (B,TF,K)
    embedding = embedding.view(Nb,-1,self.K)

    return embedding
