import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from PIL import Image



#im = Image.open('C:/Users/Mcastiblanco/Documents/AGPC/DataScience2020/Streamlit/Arroz/apps/arroz.png')
im2 = Image.open('p1.jpg')
st.set_page_config(page_title='Clas-App', layout="wide", page_icon=im2)
st.set_option('deprecation.showPyplotGlobalUse', False)

row1_1, row1_2 = st.columns((2, 3))

with row1_1:
    image = Image.open('p1.jpg')
    st.image(image, use_column_width=True)
    st.markdown('Web App by [Manuel Castiblanco](http://ia.smartecorganic.com.co/index.php/contact/)')
with row1_2:
    st.write("""
    # Clasificación App
    Esta App utiliza algoritmos de Machine Learning  para clasificar !
    """)
    with st.expander("Contact us 👉"):
        with st.form(key='contact', clear_on_submit=True):
            name = st.text_input('Name')
            mail = st.text_input('Email')
            q = st.text_area("Query")

            submit_button = st.form_submit_button(label='Send')
            if submit_button:
                subject = 'Consulta'
                to = 'macs1251@hotmail.com'
                sender = 'macs1251@hotmail.com'
                smtpserver = smtplib.SMTP("smtp-mail.outlook.com", 587)
                user = 'macs1251@hotmail.com'
                password = '1251macs'
                smtpserver.ehlo()
                smtpserver.starttls()
                smtpserver.ehlo()
                smtpserver.login(user, password)
                header = 'To:' + to + '\n' + 'From: ' + sender + '\n' + 'Subject:' + subject + '\n'
                message = header + '\n' + name + '\n' + mail + '\n' + q
                smtpserver.sendmail(sender, to, message)
                smtpserver.close()

st.header('Aplicación')
st.markdown('____________________________________________________________________')
app_des = st.expander('Descripción App')
with app_des:
    st.write("""Esta aplicación muestra a que especie de Pinguinos de Palmer de acuerdo a los parámetros""")

st.sidebar.header('Parámetros de Entrada Usario')

# st.sidebar.markdown("""
# [Example CSV input file](penguins_example.csv)
# """)

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Cargue sus parámetros desde un archivo CSV", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Isla',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sexo',('Macho','Hembra'))
        bill_length_mm = st.sidebar.slider('Longitud Pico (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Profundidad Pico (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Longitud de Aleta (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Masa Corporal (g)', 2700.0,6300.0,4207.0)
        data = {'Isla': island,
                'Pico_longitud_mm': bill_length_mm,
                'Pico_Profundidad_mm': bill_depth_mm,
                'Aleta_longitud_mm': flipper_length_mm,
                'Masa_Corporal_g': body_mass_g,
                'sexo': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['Especies'], axis=1)
df = pd.concat([input_df,penguins],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['sexo','Isla']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('Parámetros de Entrada')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('A la espera de que se cargue el archivo CSV. Actualmente usando parámetros de entrada de ejemplo (que se muestran a continuación).')
    st.dataframe(df)

# Reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

row1_1, row1_2 = st.columns((2,2))
with row1_1:
    st.subheader('Prediction')
    penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
    if prediction==0:
        image = Image.open('a.png')
        st.image(image, use_column_width=True)
        st.write(str(penguins_species[prediction]))
    elif prediction==2:
        image = Image.open('g.png')
        st.image(image, use_column_width=True)
        st.write(str(penguins_species[prediction]))
    else:
        image = Image.open('c.png')
        st.image(image, use_column_width=True)
        st.write(str(penguins_species[prediction]))
    #st.markdown('Web App by [Manuel Castiblanco](https://github.com/mcastiblanco1251)')
with row1_2:
    st.subheader('Prediction Probability')
    st.write(prediction_proba)

with st.expander("Contáctanos👉"):
    st.subheader('Quieres conocer mas de IA, ML o DL 👉[contactanos!!](http://ia.smartecorganic.com.co/index.php/contact/)')
