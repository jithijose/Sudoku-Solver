import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

def load_model():
    ''' function to load saved model '''
    
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model/model.h5")
    #print("Loaded saved model from disk.")
    
    return loaded_model


def predict_numbers(digit, model):
    ''' function to predict the number from the digit image '''
    
    number = np.argmax(model.predict(digit , verbose = 0))
    
    return number


def extract_numbers(image):
    image = cv2.resize(image, (450, 450))
    number_grid = np.zeros((9, 9))
    
    model = load_model()
    
    for i in range(9):
        for j in range(9):
            digit = image[i * 50 : (i +1) * 50, j * 50 : (j + 1) * 50]
            #cv2.imwrite(f'images/{i}_{j}.jpg', digit)
            
            if digit.sum() > 80000:
                #print(f'digit sum: {digit.sum()}')
                digit =  cv2.resize(digit, (28, 28))
                digit = digit.reshape(1, 28, 28, 1)
                prediction = predict_numbers(digit, model)
                number_grid[i][j] = prediction
            else:
                number_grid[i][j] = 0
    
    return number_grid
