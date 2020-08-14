# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 23:06:29 2020

@author: patel
"""
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import json
import re
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
app = Flask(__name__)
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

# Load Keras model
# Create the architecture of the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(215723, 100, input_length=2441),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.load_weights('Model/Imdb_model_weights.h5') # Load the weights


# Create a tokenizer from json object
with open('Model/tokenizer_oov.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

# Function to remove all the html tags from test
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

# Function to remove all the numbers from text
def remove_numbers(text):
    text1 = re.sub(r'\d+', '', text)
    return text1

# Function to remove punctuations and symbols from test
def remove_symbol(text):
    for char in string.punctuation:
        text = text.replace(char,'')
    return text

# In this processs lets remove all symbols, html tags, numbers and lower case all the words
def clean_data(text):
	# remove html tags
	temp = cleanhtml(text)
	
	# remove numbers
	temp = remove_numbers(temp)

    # remove punctuations
	temp = remove_symbol(temp)
	
	# remove all the numbers
	temp1 = []
	for i in temp.split():
		if not i.isdigit():
			temp1.append(i)
	
	# join the list
	temp = " ".join(temp1)
	
	# convert all the words into lower case
	temp = str(temp).lower()
	  
	# remove nan
	temp2= [x for x in temp.split() if str(x) != 'nan']
	
	return temp2	   




@app.route("/predict", methods=['POST'])
def predict():
	if request.method=='POST':
		text = str(request.form['Review'])
		
		# Clean the text 
		text = clean_data(text)
		
		# Convert text to sequences
		text_sequence_token = tokenizer.texts_to_sequences([text])
		
		# Add padding to tokenizer, we can extract maxlen from model.summary()
		text_pad_sequence = pad_sequences(text_sequence_token, maxlen=2441)
		
		# predict the text
		result = model.predict(x=text_pad_sequence)
		
		# Check if the sentiment is Positive, Neutral or Negative
		if result[0][0] < np.float32(0.45):
			sentiment = 'Negative'
			str1 = "The sentiment of the review is "+sentiment + " --> " + str(result[0][0])
			return render_template('index.html', prediction_texts=str1)
		elif np.float32(0.45) <=result[0][0] < np.float32(0.57):
			sentiment = 'Neutral'
			str1 = "The sentiment of the review is "+sentiment + " --> " + str(result[0][0])
			return render_template('index.html', prediction_texts=str1)
			
		elif np.float32(0.57) <= result[0][0] <= np.float32(1.0):
			sentiment = 'Positive'
			str1 = "The sentiment of the review is "+sentiment + " --> " + str(result[0][0])
			return render_template('index.html', prediction_texts=str1)
	
		else:
			return render_template("index.html")

if __name__=="__main__":
	app.run(debug=True)