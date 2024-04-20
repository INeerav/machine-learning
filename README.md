# machine-learning
The dojo work on the computer vision, multiclass/multilabel classifications, multinomial regressions, DT, RF

(sorry can`t share the dataset)

## Text classification steps

Vectorization : TfIdf vector (can choose Count vector too, depending on the problem)
Transformer : TfIdf transformer (for the extra normalisation step)
Classifier : Multinomial LR (softmax), MultiNB 

1. Scoring the comments and dataset formation
2. Validity of the sentiment analysis scores
3. Pre-processing of the dataset.
4. K-fold training
5. Grid search for best parameters
6. Testing other architectures


## Multimodal image classification, multi class steps

1. Arranging Images: Organize your dataset into separate
folders for training, validation, and testing.
2. Data Augmentation: Apply data augmentation techniques
such as rotation, flipping, and zooming to increase the
diversity of your training data.
3. Resize and Reshape: Resize images to match the input
size required by the EfficientNet model and reshape them
into the appropriate format (e.g., height, width, channels).
4. Batch Size: Determine an appropriate batch size for
training the model based on your available computational
resources.
5. One-Hot Encoding: Convert class categories into one-hot
encoded vectors to represent the target labels.
6. Calculating Class Weights: Calculate class weights to
address the imbalance in the dataset, ensuring that the
model pays more attention to minority classes during
training.
7. . Parameter Tuning: Tune hyperparameters such as learning rate, optimizer (e.g., Adam), activation function
(ReLU), and loss function (categorical crossentropy).
8. Stop Trainable Layers: Decide whether to freeze certain
layers (e.g., convolutional base) or allow them to be
trainable during training.
9. Define Callbacks: Set up callbacks such as early stopping
to prevent overfitting and save the best model weights
during training.
10. Evaluation Metrics: Choose evaluation metrics such as
accuracy, precision, recall, and F1-score to assess the
model’s performance.
11. Training the Model: Train the EfficientNet model on
the training data using the defined hyperparameters, callbacks, and class weights.
12. Predicting Image Categories: Use the trained model to
predict the categories of images in the validation and test
datasets

