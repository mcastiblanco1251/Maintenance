import pandas as pd
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_2P, Fit_Weibull_3P
from reliability.Other_functions import crosshairs
from reliability.Fitters import Fit_Everything
from reliability.Other_functions import make_right_censored_data
from reliability.Probability_plotting import plot_points
from pandas.io.formats.style import Styler
import glob
from pandas import ExcelWriter
from datetime import date
import streamlit as st
from PIL import Image


im = Image.open("mtto2.png")

st.set_page_config(page_title='Confiabilidad', layout="wide", page_icon=im)
st.set_option('deprecation.showPyplotGlobalUse', False)

row1_1, row1_2 = st.columns((2,3))

with row1_1:
    image = Image.open('mttoch.jpg')
    st.image(image, use_column_width=True)
    st.markdown('Web App by [Manuel Castiblanco](http://ia.smartecorganic.com.co/index.php/contact/)')
with row1_2:
    st.write("""
    # Confiabilidad App
    Esta app usa machine learning  para predecir la confiabilidad de los equipos en base a sus tiempos de falla!
    """)
    with st.expander("Contact us 👉"):
        with st.form(key='contact', clear_on_submit=True):
            name=st.text_input('Nombre')
            mail = st.text_input('Email')
            q=st.text_area("Consulta")

            submit_button = st.form_submit_button(label='Enviar')
            if submit_button:
                subject = 'Consulta'
                to = 'macs1251@hotmail.com'
                sender = 'macs1251@hotmail.com'
                smtpserver = smtplib.SMTP("smtp-mail.outlook.com",587)
                user = 'macs1251@hotmail.com'
                password = '1251macs'
                smtpserver.ehlo()
                smtpserver.starttls()
                smtpserver.ehlo()
                smtpserver.login(user, password)
                header = 'To:' + to + '\n' + 'From: ' + sender + '\n' + 'Subject:' + subject + '\n'
                message = header + '\n'+name + '\n'+mail+'\n'+ q
                smtpserver.sendmail(sender, to, message)
                smtpserver.close()


st.header('Aplicación')
st.write('_______________________________________________________________________________________________________')

app_des=st.expander('Descripción de la App')
with app_des:
    st.markdown("""
    Esta app esta desarollada para poder predecir la confiabilidad de los equipos industriales usando Machine Learning y
    modelos estadísticos.

    Tener en cuenta:

    * **Dirección**: Es donde se encuentra los archivos con la información, esta debe estar de acuerdo a la plantilla
    para que funcione adecuadamente, si esta en su PC use C:/xxx/xxx/ o su esta en la URL de streamlit use ./Demo; para mayor información contactar [Manuel Castiblanco](http://ia.smartecorganic.com.co/index.php/contact/)
    * **Extensión**: Hace referencia al tipo de extensión de los archivos para hacer el análisis esta debe estar de acuerdo a la plantilla
    para que funcione adecuadamente, para mayor información contactar [Manuel Castiblanco](http://ia.smartecorganic.com.co/index.php/contact/)
        """)
#Necesario1
def falla_tiempo(df):
    df['Fecha']=pd.to_datetime(df['Fecha'], dayfirst=True)#infer_datetime_format=True)
    df['Falla_tiempo']=df['Fecha'].diff(periods=1)
    df['Falla_tiempo']=df['Falla_tiempo'].fillna(pd.Timedelta(days=0))
    df['Falla_tiempo'] = df['Falla_tiempo'].dt.days.astype('int16')
    df.loc[df['Falla_tiempo'] == 0, 'Falla_tiempo'] = 1
    return df

#Necesario2
def confiabilidad(threshold, data, wb):
    data
    time_to_p=max(data)
    time_to_p
    hours=list(np.array(range(1,time_to_p,10)))
    hours
    threshold=threshold
    realiability=[]
    for hour in hours :
        hours[0]
        sf = wb.distribution.SF((hours[0]))
        if sf<=threshold:
            break
    st.write(f"Horas para que NO falle {hour} horas o {round(hour/24,2)} dias, con una confiabilidad del {round(sf,2)*100}% o probibilidad de falla de {round((1-sf),2)*100}%")
    hour=hour
    sf=sf
    p=1-sf
    return hour, round(sf,2), round(p,2)


#Necesario3
def weibull_curvas(data, equi):
    #ar=equi.split('.')[0]
    #st.subheader(f'Curvas de Confiabilidad para equipo {ar}')
    #data = data # made using Weibull Distribution(alpha=50,beta=3)
    wb = Fit_Weibull_2P(failures=data)
    #fig=plt.figure(figsize=(6,5))
    st.pyplot(plt.show())
    wb.distribution.SF(label='Fitted Distribution',color='steelblue')
    plot_points(failures=data,func='SF',label='failure data',color='red',alpha=0.7)
    plt.legend()
    st.pyplot(plt.show())
    wb.distribution.HF(label='Fitted Distribution',color='steelblue')
    plot_points(failures=data,func='HF',label='failure data',color='red',alpha=0.7)
    plt.legend()
    st.pyplot(plt.show())
    return wb

    #Necesario4
def pareto_componente(df):
    comp_fail=df[' COMPONENTE '].value_counts()
    comp_fail
    #comp_fail.plot.bar(y=df[' COMPONENTE '].value_counts())
    #fig=plt.figure(figsize=(6,5))
    plt.title('Pareto Componentes')
    plt.bar (comp_fail.index, comp_fail.values)
    plt.xlabel('Componente',fontsize=9)
    plt.xticks(rotation=90)
    plt.ylabel('Numero de veces',fontsize=9)
    st.pyplot(plt.show())

    st.pyplot()
    #Necesario5
def componentes(df):
    a=df.groupby(' COMPONENTE ').count()
    a=a[a['Falla_tiempo']>=3]
    a=a.index
    componentes=np.array(a)
    #componentes=comp2(componentes)
    return componentes

    #Necesario 6
def comp2(componentes):
    for componente in componentes:
        filt=(df[' COMPONENTE ']==componente)
        dfC=df[filt]
        datac=falla_tiempo_comp(dfC)
        datac=datac['Falla_tiempo'][1:]
        datac=np.array(datac)*24
        #datac
        if np.all(datac == datac[0]):
            print('All Values in Array are same / equal')
            index = np.argwhere(componentes=='COMP 4')
            componentes=np.delete(componentes, index)
    return componentes

def falla_tiempo_comp(df):
    df['Fecha']=pd.to_datetime(df['Fecha'],dayfirst=True)#, infer_datetime_format=True)
    df['Falla_tiempo']=df['Fecha'].diff(periods=1)
    df['Falla_tiempo']=df['Falla_tiempo'].fillna(pd.Timedelta(days=0))
    df['Falla_tiempo'] = df['Falla_tiempo'].dt.days.astype('int16')
    df.loc[df['Falla_tiempo'] == 0, 'Falla_tiempo'] = 1
    return df

#Necesario 6
def weibull_componentes( df, threshold, equi):
    time_to_p=max(data)
    hours=np.array(range(1,time_to_p,10))
    threshold=threshold
    alpha=[]
    beta=[]
    hours_=[]
    reliability=[]
    fail=[]
    days=[]
    for componente in componentes:
        st.subheader(f"Análisis para el {componente} del equipo {equi.split('.')[0]}")
        filt=(df[' COMPONENTE ']==componente)
        dfC=df[filt]

        datac=falla_tiempo_comp(dfC)
        datac=datac['Falla_tiempo'][1:]
        datac=np.array(datac)*24

        #wbc=Fit_Weibull_2P(failures=datac)
        wbc=weibull_curvas(datac, equi)

        alphac=wbc.results['Point Estimate'][0]
        betac=wbc.results['Point Estimate'][1]
        alpha.append(alphac)
        beta.append(betac)
        for hour in hours :
            sf = wb.distribution.SF(hour)
            if sf<=threshold:
                break
        hours_.append(hour)
        reliability.append(round(sf,2))
        fail.append(round((1-sf),2))
        days.append(round(hour/24,2))
        st.subheader(f"Confiabilidad para {componente} del equipo {equi.split('.')[0]}")
        st.write(f"Horas para que el componente {componente} NO falle {hour} horas o {round(hour/24,2)} dias, con una confiabilidad del {round(sf,2)*100}% o probibilidad de falla de {round((1-sf),2)*100}%", '\n')
    table={'Equipo':equi.split('.')[0],'componente':componentes,'alpha':alpha,'beta':beta, 'hours':hours_, 'days': days, 'reliability':reliability, 'fail':fail}
    table=pd.DataFrame.from_dict(table)
    return table#.style.set_caption(f'Tabla Resumen Análiis de Confiabilidad {equi}')#print(f'Tabla resumen de {equi}''\n',f'y la siguiente Tabla: {table}','\n')

def files(pa, ext):
    extension = ext
    os.chdir(pa)
    files = glob.glob('*.{}'.format(extension))
    return files

#with st.form(key='Descargar', clear_on_submit=False):
st.sidebar.subheader('Datos Entrada')
#try:
path='SELLADORA PEQUEÑA.xlsx'#st.sidebar.text_input('Dirección')#'Y:/2016 MANTENIMIENTO/confiabilidad/'
ext=st.sidebar.selectbox('Extensión archivo', ('xlsx','csv'))
tr=st.sidebar.slider('Confiabilidad',0.0,1.0,0.55)
if path=='':
    st.sidebar.error('Favor introducir la Dirección para hacer el análisis, (*Ver descripción de la app*)')
else:

    #    submit_button = st.form_submit_button(label='Analizar')
    #    if submit_button:
    files=path

    st.subheader('Archivos a Analizar')
    #f=pd.DataFrame(files, columns=['Equipos'])
    st.write(path)#=['MASTER 35.csv']
    # except:
    # pass

    #
    #try:
    #df1=pd.read_excel(path+files[0])#, sep=';')#,encoding='latin-1',sep=';')
    #st.table(df)
    threshold=tr

    alpha_g=[]
    beta_g=[]
    tables=[]
    conf_g={}


    for file in files:
        equi=file#'Vitrojet .csv'
        df=pd.read_excel(path)#, sep=';')#,encoding='latin-1',sep=';')#,encoding='ISO-8859-1')
        #Falla general equipo
        df=falla_tiempo(df)


        #Analisis general de equipo
        data=df['Falla_tiempo'][1:]
        data=np.array(data)*24
        st.subheader(f"Curvas de Confiabilidad para equipo {equi.split('.')[0]}")
        wb=weibull_curvas(data, equi)
        alpha_g=wb.results['Point Estimate'][0]
        beta_g=wb.results['Point Estimate'][1]
        st.subheader(f"Confiabilidad Global del equipo {equi.split('.')[0]}")
        hour, sf, p= confiabilidad(threshold,data, wb)
        conf_g[file]={'alpha':alpha_g,'beta':beta_g, 'hours':hour, 'days': hour/24, 'reliability':sf, 'fail':p}

        #pareto componente
        st.subheader(f"Pareto de Falla por Componentes del equipo {equi.split('.')[0]}")
        pareto_componente(df)
        #Analisis componente

        a=df.groupby(' COMPONENTE ').count()
        a=a[a['Falla_tiempo']>=3]
        a=a.index
        componentes=np.array(a)
        componentes=comp2(componentes)
        #componentes=componentes(df)
        table=weibull_componentes( df, threshold,equi)
        tables.append(table)
    conf_g=pd.DataFrame.from_dict(conf_g)
    #tables
    st.subheader('Tabla Resultados por Equipo')
    table={}
    for i in range(len(tables)):
        table[i]=pd.DataFrame(tables[i])
        table[i]

    #Descarga archivo Global de Equipos
    # st.subheader('Descargar Reporte')
    # import time
    # with st.spinner('Generando reporte en excel...'):
    #     time.sleep(5)
    # st.success('Hecho!')
    # writer = ExcelWriter(path+f'Confiabilidad_Global_Equipos_Críticos{date.today()}'+'.xlsx')
    # conf_g.T.to_excel(writer,'Conf_Global_Equipos_Críticos')
    # writer.save()
    # writer = ExcelWriter(path+f'Confiabilidad_Equipos_Críticos_Componente{date.today()}'+'.xlsx')
    # for i in range(len(tables)):
    #     tables[i].to_excel(writer,f'{conf_g.T.index[i]}')
    # writer.save()
