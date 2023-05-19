import torch
from torch.utils.data import Dataset, DataLoader,random_split
from torchaudio.transforms import Spectrogram
import numpy as np
import random

class RandomClip:
  def __init__(self, fraction):
    self.fraction = fraction
    

  def __call__(self, audio_data):
    # Audio datas are stored in list
    
    audio_length = audio_data[0].shape[0]
    clip_length = int(self.fraction * audio_length)
    if audio_length > clip_length:
      offset = random.randint(0, audio_length-clip_length)
      audio_data = (x[offset:(offset+clip_length)] for x in audio_data)
    return audio_data

def norm(x):
  return (x - np.mean(x))/np.std(x)
     

class mus_dataset(Dataset):

  def __init__(self,DB,n_tracks):

    # We will keep things binary - Vocal vs Drum (single channel)

    super(mus_dataset,self).__init__()
    self.DB = DB[:n_tracks]
    self.fs = DB[0].rate
    self.transform = RandomClip(0.3)

    # Step 1 : Extract time domain vocal and drum (Normalize)
    self.S = []

    for i in range(n_tracks):
      # Extract the single channel data
      vocals = np.mean(self.DB[i].targets['vocals'].audio,axis = 1)
      drums = np.mean(self.DB[i].targets['drums'].audio,axis = 1)

      # normalize the data and append as tuple
      vocals = norm(vocals)
      drums = norm(drums)

      self.S.append((vocals,drums))


  def __len__(self):
    return len(self.S)

  def __getitem__(self,i):
    
    # Get current source and crop
    V,D = self.S[i]
    V,D = self.transform((V,D))

    # Get a tensor for source
    V = torch.Tensor(V)
    D = torch.Tensor(D)

    # Individual Sources
    X = torch.stack([V,D],0)
    Y = V + D

    return X,Y,self.fs

    
    
