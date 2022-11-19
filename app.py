import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np
from PIL import Image
model = pickle.load(open('model.sav', 'rb'))

st.title('Salary Prediction')
st.sidebar.header('employee Data')
image = Image.open('salary.PNG')
st.image(image, '')
image2 = Image.open('title.png')
st.image(image2, 'titles')
image3 = Image.open('encode.png')
st.image(image3, 'title encoding')

# FUNCTION
def user_report():
  title = st.sidebar.slider('title', 0,14, 1 )
  exp_years = st.sidebar.slider('exp_years', 0,60, 1 )


  user_report_data = {
      'title':title,
      'exp_years':exp_years
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

user_data = user_report()
st.header('Employee Data')
st.write(user_data)

salary = model.predict(user_data)
st.subheader('Employee Salary')
st.subheader(str(np.round(salary[0], 2))+'$')
