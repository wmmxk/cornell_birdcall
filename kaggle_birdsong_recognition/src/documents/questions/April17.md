1. In the winning solution, not all 5-second clip is predicted. Only the clip that has a bird call
   is generated in the submission file. Although na is filled by "nocall" in 
   test/inference-cornell-birdcall-identification-1st-place-solution.py:1065. In fact there is no na.
   Is your understanding correct?
   
2. Why does each frame represent a 5-second clip after the raw audio [1, 961000] is converted to [3001, 264]   