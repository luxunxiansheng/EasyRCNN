1. Following the guide to organize the project: <https://julienbeaulieu.github.io/2020/03/16/building-a-flexible-configuration-system-for-deep-learning-models/>

2. pytorch engineering guide <https://github.com/IgorSusmelj/pytorch-styleguide>



3. It is anti-pattern to use the __Call__ method in model classe to call the forward method. A better way is to use a explicit predict method to call the forward method and  return the output. It is easier for the user to know it is a prediction method.


4. Loss class shoudl provide an explicit method, say , compute(), to comupte the loss and return the result instead of using the __call__ method.  
   

