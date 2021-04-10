1. An essential question is to how predict segmentwise prediction if the label is clipwise.
   This question is faced by all the public solution, Besides, the winning solution is based on 
   another solution, so roll back, look into that one.
   
   Answer: (source, https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection?select=best.pth)
    In weakly-supervised setting, we only have clip-level annotation, 
    therefore we also need to aggregate that in time axis. 
    Hense, we at first put classifier that outputs class existence probability 
    for each time step just after the feature extractor and 
    then aggregate the output of the classifier result in time axis. 
    In this way we can get both clip-level prediction 
    and segment-level prediction (if the time resolution is high,
     it can be treated as event-level prediction). 
    Then we train it normally by using BCE loss with clip-level prediction and clip-level annotation.
    
2. Is birdclef competition the same as the cornell one? If so, find the weight of the inference model
   of the winning solution last time. 