


class Perceptron:
  #eta - learning rate the scalar value

  def __init__(self, eta, epochs):
    self.weights = np.random.randn(3) * 1e-4 # 3 for 3 inputs
    print(f"intial weights before training: {self.weights}")
    self.eta = eta
    self.epochs = epochs

  def activationFunction(self, inputs, weights):   #sigmoid function
    z= np.dot(inputs,weights) # z = W * X
    return np.where(z > 0, 1, 0)
  #fit method training steps

  def fit(self,X,y):
    self.X = X
    self.y = y
  # self can be used any wher ein the class a global variable
    X_with_bias = np.c_[self.X,-np.ones((len(self.X),1))]
    print(f"X with bias :\n {X_with_bias}")
    for epoch in range(self.epochs):
      print("--"*10)
      print(f"for eposch : {epoch}")
      print("--"*10)
      #forward propagation 
      y_hat = self.activationFunction(X_with_bias,self.weights)
      print(f"predicted value after forward pass: {y_hat}")
      self.error = self.y - y_hat;
      print(f"error : \n {self.error}")
      self.weights = self.weights + self.eta * np.dot(X_with_bias.T,self.error)
      print(f"Updated weights after epoch: {epoch}/{self.epochs},weights : {self.weights}")
      print("#####"*10)

  # prediction
  def predict(self,X):
    X_with_bias = np.c_[X, -np.ones((len(X),1))]
    return self.activationFunction(X_with_bias, self.weights)
  
  def prepare_data(df):
    X = df.drop("y",axis=1)
    y = df["y"]
    return X,y
  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f"total loss: {total_loss}")
    return total_loss




