April 2:
1. On which dimension, conv1d works? documents/operations_data_in_model.md:9
2. how avg_poo1d works? documents/operations_data_in_model.md:24
3. why transpose the input before the fc layer? documents/operations_data_in_model.md:25
4. Is the first column denotes no birdcall? kaggle-birdsong-recognition/src/test/cornell-birdcall-identification-1st-place-solution.py:953
5. you don't understand global_time, kaggle-birdsong-recognition/src/test/cornell-birdcall-identification-1st-place-solution.py:971
6. what is going on here: kaggle-birdsong-recognition/src/test/cornell-birdcall-identification-1st-place-solution.py:1028
    int((onset // 5) * 5) + 5 means round onset to a multiple of 5. e.g if onset = 4, it is rounded to 5.
7. what does cur_section mean, kaggle-birdsong-recognition/src/test/cornell-birdcall-identification-1st-place-solution.py:1039
8. How comes the 5? kaggle-birdsong-recognition/src/test/cornell-birdcall-identification-1st-place-solution.py:1028
   5 means 5 seconds. See the Thresholds section in this https://www.kaggle.com/c/birdsong-recognition/discussion/183208
