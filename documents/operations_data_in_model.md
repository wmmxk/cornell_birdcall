preprocessing 
- extract spectrogram
  The essence is to do conv1d on the raw audio. 
  -> raw audio: [815616, ]
  -> create a copy of a tensor of the fixed length, and embed the raw audio in the copy. The purpose is to unify the length. [1, 960000].
  -> makes 10 copies  [10, 961000]
  -> pad by zeros on two ends of each audio. [10, 1, 961024]
  -> apply conv1d on the raw audios, two pass, one for real, one for imaginary.
     conv1d: (1, 513, kernel_size=(1024, ), stride=(320, ), bias=True)
     [10, 513, 3001], [10, 513, 3001]
  -> add a dimension and transpose:
     [10, 1, 3001, 513]
- extract logmel    
  -> matrix multiplication, [10, 1, 3001, 64].
  -> torch.log10, [10, 1, 3001, 64]