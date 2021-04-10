1. Which part is different between training and inference?
   What are the loss for clipwise prediction and framewise prediction?
2. How does the clipwise prediction and framewise prediction reconcile?
3. Why does the parallel.run framework allow you step into? Probably you need to understand 
   ignite.distributed first. This part is separate from the SedEngine part. Maybe you can 
   just run the the run() function
   
4. You can not check a variable in the console. Extract the dataloader and model, run one-step of tranining
   kaggle-birdsong-recognition/src/engine/base/base_engine.py:240
   prepare_batch: kaggle-birdsong-recognition/src/engine/sed_engine.py:17
        engine.state.output = None
        self.model.train()
        x, y = self.prepare_batch(batch, mode = 'train')
        y_pred = self.model(x)
        loss, dict_loss = self.loss_fn(y_pred, y)
        self.loss_backpass(loss).
        
5. Try VScode, if it allows you to look into the code, no need to break it down.        