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


class kmeans(nn.Module):

  def __init__(self,Nc,K):

    # K is dimension and Nc is number of clusters
    super(kmeans,self).__init__()

    self.Nc = Nc
    self.K  = K

    self.mu = torch.stack([torch.rand(1,K) for i in range(self.Nc)],dim=0), # We make it Nc, 1 by K for consistency

  def fit(self,X,niter):

    # We will need to flatten the X vector regardless in order to make sure we work not with batches anymore

    X = X.view(1,-1,self.K) # 1,Ns,K
    
    for t in range(niter):

      # Perform assignment
      Dist = torch.sqrt(torch.sum((X-self.mu)**2,-1)) # Nc,Ns

      # Return argmax of this
      cluster_idx = torch.argmin(Dist,dim=0) # A vector of menmbership (Ns,)
 
      # Reevaluate the cluster center
      for c in range(self.Nc):
        
          # Find the binary mask for each class
          bool_mask = torch.stack([cluster_idx==c for _ in range(self.Nc)],-1)
          subdata = X[bool_mask]
          mu_c = torch.mean(subdata,dim=0).view(1,-1) # 1 by K
          self.mu[c,:,:] = mu_c
      
      # repeat this process for number of iterations








