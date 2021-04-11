- 1.preprocessing 
  - extract spectrogram, with learnable parameters
    - load audio and standardize the len
      -> raw audio: [815616, ]
      -> create a copy of a tensor of the fixed length, and embed the raw audio in the copy. The purpose is to unify the length. [1, 960000].
      -> makes 10 copies  [10, 961000]
    - STFT
      The essence is to do conv0d on the raw audio. 
      -> pad by zeros on two ends of each audio. pad len is n_fft/2 on each side, [10, 1, 961024]
      -> apply conv1d on the raw audios, two pass, one for real, one for imaginary.
         conv1d: (1, 513, kernel_size=(1024, ), stride=(320, ), bias=True)
         [10, 513, 3001], [10, 513, 3001]
      -> add a dimension and transpose: [batch, 1, time len/frame number, multiple filters/channel]
         [10, 1, 3001, 513] 
  - extract logmel, not learnable    
    -> matrix multiplication, meaning map the 513 channels to 64 channels. [10, 1, 3001, 64].
    -> torch.log10, [10, 1, 3001, 64]
    -> transpose (1, 3), [10, 64, 3001, 1] then do a BatchNorm2d. For image task, the input is [N, C, H, W]
    -> transpose back, [10, 1, 3001, 64]
    -> time_dropper, freq_dropper, only apply on training data.
    -> expand the 2nd dimension, [10, 3, 3001, 64], Up to now, you can think of each audio has been converted to an image of three repeated channels.
                                                    3001 is frame, 64 is frequency.
- 2.cnn_feature_extractor
    -> a densenet121, [10, 1024, 93, 2]
    -> aggregate in frequency axis: [10, 1024, 93]
    -> max_pool1d, and avg_pool, sum them up, [10, 1024, 93], 
    -> dropout, [10, 1024, 93]
    -> transpose, [10, 93, 1024]
    -> fully connection layer: [10, 93, 1024]
    -> transpose back: [10, 1024, 93]
- 3.attention block
    -> norm_att, attention layer is a conv1d layer+softmax, attention is along the last dimension, which is frame, 264 is then number of class, [10, 264, 93]
    -> cla, the same input to attention is passed to conv1d+sigmoid, [10, 264, 93]
       You can treat the 264 dimensions separately, each dimension is a probability of each segment falling in this class.
    -> norm_att * cla, x, the sum over the last dimension, [10, 264], sum over probability over all segments.
       The essence is to build clipwise output from segmentwise output.
    return x (clipwise output), norm_att, cla (segmentwise output)    
    cla -> transpose, cla, [10, 93, 264] [batch_size, time_steps, classes_num]
    
-4. interpolate, padding ,transpose
    - segmentwise, framewise
      cla -> interpolate, framewise output [10, 2976, 264], 2976 = time_steps (93) * ratio (32)
      framewise ouput -> padding to the origanl length after preprocessing,  [10, 3001, 264]
      reshape -> [1, 10, 3001, 264]
    - clipwise
      reshape -> [1, 10, 264]

-5  extract raw framewise prediction, and save clipwise prediction
    - framewise outputs
      average over batch, [3001, 264], the first dimension is 1, does not matter
      compare it with threshold, generating a binary outputs, [3001, 264]
      analyze each column using a while loop, 
        - np.argwhere(), returns a list of indices (detected) of True, each index represents a frame
        - merge adjacent true indices:
          - head_idx and tail_idx both start from 0
          - check detected[tail_idx+1] - detected[tail_idx] == 1 or not.
            - if not, means a segment ends
              - extract a segment
                - onset_idx = detected[head_idx]
                - offset_idx = detected[tail_idx]
                - onset = 0.01 * onset_idx + global_time
                - offset = 0.01 * offset_idx + global_time
                - get the max and mean confidence in that interval framewise_outputs[onset_idx:offset_idx, target_idx]
              - move the head_idx and tail_idx, both starts with the last tail_idx
                - head_idx = tail_idx + 1
                - tail_idx = tail_idx + 1
            if yes, means the previous segment continues
              - tail_idx += 1
        - if head_idx > len(detected), break the while loop

-6 post-processing.
   Assume there is only one model
   - it is possible that there are multiple birdcalls at one point, so it is set, 
        each element is is indexed by row_id = f"{site}_{audio_id}_{start_section}"
     kaggle-birdsong-recognition/src/test/cornell-birdcall-identification-1st-place-solution.py:1032
     Short event:
         If a birdcall lasts less than 5 seconds, because of the rounding, start_section == end_section, in this case.
         In this case, the event is denote by the start_section with site and audio id
     Long event: end_section - start_section
         If the event is longer than 5 seconds, multiple events are recorded, one by start_section, others by 
           incrementing the cur_section by 5 every time, until, cur_section == end_section. 
         
   - groupby all the predictions by a single model by audio id, extract all the birdcall events
        
  
    - clipwise outputs
      average over batch, [264, 1]
      compare it with clip_threshold (0.3), generating a binary outputs [264, 1]
      np.argwhere(), returns the indices of all Trues, each index represents a type of bird
      
      
   
      
    
    
  
  

