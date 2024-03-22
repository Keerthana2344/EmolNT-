# EmolNT-
Identify the intensity of emotions from the given text
Here I have used the training and testing dataset for emotions on ANGER. Similar code can be used for various emotions. Existing emotion datasets are mainly annotated categorically without an indication of degree of emotion. Further, the tasks are almost always framed as classification tasks (identify 1 among n emotions for this sentence). In contrast, it is often useful for applications to know the degree to which an emotion is expressed in text. This is the first task where systems have to automatically determine the intensity of emotions in tweets.
TASK: Given a tweet and an emotion X, determine the intensity or degree of emotion X felt by the speaker -- a real-valued score between 0 and 1. The maximum possible score 1 stands for feeling the maximum amount of emotion X (or having a mental state maximally inclined towards feeling emotion X). The minimum possible score 0 stands for feeling the least amount of emotion X (or having a mental state maximally away from feeling emotion X). The tweet along with the emotion X will be referred to as an instance. 

APPROACH: In the models directory, I have created a python code that includes both machine learning model with the best accuracy - Ridge Regression and in deep learning I have used LSTM model to solve the task.


