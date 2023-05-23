from os import X_OK
import torch.nn as nn
import torch
from torchaudio.transforms import Spectrogram,InverseSpectrogram
import DANet.data as data
import numpy as np
from tqdm import tqdm


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

  def __init__(self,Nc,K,X):

    # K is dimension and Nc is number of clusters
    super(kmeans,self).__init__()

    self.Nc = Nc
    self.K  = K
    self.X = X

    self.mu = torch.stack([self.X[torch.randint(0,X.shape[0],(1,))] for i in range(self.Nc)],dim=0) # We make it Nc, 1 by K for consistency

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
          X = X.squeeze(0)
          bool_mask = torch.stack([cluster_idx==c for _ in range(self.K)],-1)
          subdata = X[bool_mask].view(-1,self.K)
          mu_c = torch.mean(subdata,dim=0).view(1,-1) # 1 by K
          self.mu[c,:,:] = mu_c
      
      # repeat this process for number of iterations

      print(f'Iter{t}:',self.mu)

  def label(self,X):

    # For each data we will label 0 --- K - 1 for each data given
    # Return tensor (Ns,1) labels

    Dist = torch.sqrt(torch.sum((X-self.mu)**2,-1)) # Nc,Ns
    cluster_idx = torch.argmin(Dist,dim=0) # A vector of menmbership (Ns,)

    return cluster_idx

##################################################################################################

def train(embedder,trainloader,optimizer,criterion,n_epochs,batch_size,eps=1e-8):

  train_loss = []
  stft_trans = Spectrogram(n_fft=1024,onesided=True,power=None)
  inv_trans = InverseSpectrogram(n_fft=1024,onesided=True)
  for t in range(n_epochs):
    print(f"------------epoch{t+1}--------------")
    embedder.train() # train mode
    losses = []
    for x,y,fs in tqdm(trainloader):
        
        #=============== Forward Pass ====================
        losses_b = []
        optimizer.zero_grad()

        # Preprocess
        data_dict = data.preprocess(x,y,n_fft = 1024,eps = 1e-8)

        # Embedding
        X = data_dict['sm'].cuda()
        Nf = X.shape[-2]
        Nt = X.shape[-1]
        Nb = X.shape[0]
        V = embedder(X)

        # Evaluate the irms needed
        irm0 = data_dict['irm0']
        irm1 = data_dict['irm1']
        irm0 = irm0.reshape(Nb,1,-1).cuda()
        irm1 = irm1.reshape(Nb,1,-1).cuda()

        

        # Evaluate activation on each channel
        A0 = torch.bmm(irm0,V)#/torch.sum(irm0,-1) # -1 K
        A1 = torch.bmm(irm1,V)#/torch.sum(irm1,-1) # -1,K
        A  = torch.concat([A0,A1],1)

        
        
        # Evaluate Mask from the attractor points
        M = torch.bmm(A,torch.transpose(V,-1,-2))
        M = nn.Softmax(dim=1)(M)
        M = M.view(-1,2,Nf,Nt)

        #=============== Backward Pass ====================

        phase = data_dict['phase'].cuda()

        # Stacking the filtered signal
        Y = torch.stack([M[:,c,:,:]*X*torch.exp(1j*phase) for c in range(2)],1)

        # Inverse process
        y0 = inv_trans(Y[:,0,:,:])
        y1 = inv_trans(Y[:,1,:,:])

        # MSE Loss
        loss = criterion(inv_trans(stft_trans(x[:,0,:].cuda())),y0) + criterion(inv_trans(stft_trans(x[:,1,:].cuda())),y1)
        optimizer.step()

        losses_b.append(loss.item()/batch_size)
  
    losses.append(np.mean(losses_b))
    print(f'===> Epoch {t+1}: Train Loss -> {losses[-1]}')
