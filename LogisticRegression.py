import numpy as np 

def sigmoid(z):
  return 1/(1+exp(-z))
  
def forward_propagation(w, b, X, Y):
  m = X.shape[1]
  A = sigmoid(np.dot(w.T, X)+b)
  J = (-1/m)*np.sum(Y*np.log(A)+(1-y)*np.log(1-A))
  J = np.squeeze(J)
  dw = (1/m)*np.dot(X, (A-Y).T)
  db = (1/m)*np.sum(A-Y)
  
  return dw, db, J
  
def optimize(w, b, X, Y, num_iterations, alpha, print_cost = False):
  all_J = []
  for i in range(num_iterations):
    dw, db, J = forward_propagation(w, b, X, Y)
    all_J.append(J)

    w -= alpha*dw
    b -= alpha*db

  if print_cost:
    print(f'Cost at {i}th iteration: {J}')

  return w, b, all_J
  
def predict(w, b, X):
  Z = np.dot(w.T, X) + b
  A = sigmoid(Z)

  y_pred = np.zeros((1, X.shape[1]))

  for i in range(X.shape[1]):
    if A[0, i] <=0.5:
      y_pred[0, i] = 0
    else:
      y_pred[0, i] = 1
    
  return y_pred
  
def LogisticRegression(X_train, Y_train, X_test, Y_test, num_iterations = 20000, alpha = 0.5, print_cost = False):
  w = np.zeros((X_train.shape[1], 1))
  b = 0
  w, b, all_J = optimize(w, b, X_train, Y_train, num_iterations, alpha, print_cost)

  Y_pred_test = predict(w, b, X_test)
  Y_pred_train = predict(w, b, X_train)

  print(f"train accuracy: {100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100}")
  print(f"test accuracy: {100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100} ")
  
  
