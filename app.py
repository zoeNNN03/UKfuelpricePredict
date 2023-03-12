import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn import linear_model, metrics as mt
import math as m

pd.set_option('display.width', 10000)

st.markdown("<h1 style='text-align: center; color: white;'>Predict UK fuel price</h1>", unsafe_allow_html=True)

#Data Preparation
@st.cache_data(persist=True)
def load_data():
    df = pd.read_csv('./data/UKfuel.csv')
    return df
data = load_data()
data = data.drop(['Unnamed: 0'], axis = 1)

data2 = data.copy()
data2 = data2.drop( ['Duty rate in pence/litre (ULSD)'] , axis = 1)
data2 = data2.drop( ['VAT percentage rate (ULSD)'] , axis = 1)
data2.rename(columns={'Pump price in pence/litre (ULSP)' : 'Ultra-low-sulfur Petrol'},inplace=True)
data2.rename(columns={'Pump price in pence/litre (ULSD)' : 'Ultra-low-sulfur Diesel'},inplace=True)
data2.rename(columns={'Duty rate in pence/litre (ULSP)' : 'Duty rate'},inplace=True)
data2.rename(columns={'VAT percentage rate (ULSP)' : 'VAT'},inplace=True)

#Regression
data3 = data2.copy()
percent = round(data3.shape[0] * (80/100))
data_train = data3[:percent]
data_test = data3[percent:]
regr = linear_model.LinearRegression()

#predict
def predict(x1 , x2 , x3, choose, xpred):
    x = data_train[[ xpred,
           'Duty rate',
           'VAT']]
    y = data_train[choose]
    regr.fit(x , y)
    data3['Predict'] = regr.predict(data3[[ xpred,
           'Duty rate',
           'VAT']])
    return round(regr.intercept_ + (regr.coef_[0] * x1) + (regr.coef_[1] * x2) + (regr.coef_[2] * x3), 6)

st.header("Model predict data")
choose = st.radio(
     "Select the data you want to predict.",
     ('Ultra-low-sulfur Petrol', 'Ultra-low-sulfur Diesel'))

if choose == 'Ultra-low-sulfur Petrol':
    price = st.number_input("Ultra-low-sulfur Diesel", min_value = 0.00, format="%.2f")
    duty = st.number_input("Duty rate", min_value = 0.00, value = data2['Duty rate'][len(data2)-1], format="%.2f")
    vat = st.number_input("VAT", min_value = 0.00, value = data2['VAT'][len(data2)-1], format="%.2f")

    x_predict = 'Ultra-low-sulfur Diesel'

else:
    price = st.number_input("Ultra-low-sulfur Petrol", min_value = 0.00, format="%.2f")
    duty = st.number_input("Duty rate", min_value = 0.00, value = data2['Duty rate'][len(data2)-1], format="%.2f")
    vat = st.number_input("VAT", min_value = 0.00, value = data2['VAT'][len(data2)-1], format="%.2f")

    x_predict = 'Ultra-low-sulfur Petrol'

if st.button(label='SUBMIT'):
    st.subheader(f'{choose} ราคาประมาณ {predict(price, duty, vat, choose, x_predict)}')

    #ความน่าเชื้อถือของข้อมูล
    MAE = mt.mean_absolute_error(data3[choose], data3['Predict'])
    MSE = mt.mean_squared_error(data3[choose], data3['Predict'])
    RMSE = m.sqrt(MSE)
    r2 = mt.r2_score(data3[choose], data3['Predict'])
    st.write(f'MAE = {MAE : .3f} | MSE = {MSE : .3f} | RMSE = {RMSE : .3f} | r2 = {r2 : .3f}')

    pred_y = regr.predict(data_test[[ x_predict, 'Duty rate', 'VAT']])
    scat = go.Scatter(
        x = data_test[x_predict],
        y = data_test[choose],
        mode = 'markers',
        name = 'Data test')

    line = go.Scatter(
        x = data_test[x_predict],
        y = pred_y,
        name = 'Predict data', 
        marker = dict(
            color = "Red"
            ),
        mode = 'lines',
        fill = 'tonexty',
        fillcolor = 'rgba(167, 167, 167, 0.12)',
        )
    grah = [scat, line]
    layout = go.Layout(title = 'Reliability of model',
                        xaxis = dict(title='x'),
                        yaxis = dict(title='y'))           
    fig = go.Figure(data=grah, layout=layout)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.subheader(f'Enter all information, press SUBMIT')

#กราฟ
st.header("Select to display information")
grah = data.copy()
price, duty = '', ''
grah['Date'] = pd.to_datetime(grah['Date'])
select = st.selectbox("Select 1 option.", 
            ('none', 'Ultra-low-sulfur Petrol', 'Ultra-low-sulfur Diesel'),
            index = 0)

if select == 'Ultra-low-sulfur Petrol':
    price = go.Scatter(
                        x=grah['Date'],
                        y=grah['Pump price in pence/litre (ULSP)'],
                        mode='markers',
                        name='Price')
    duty = go.Scatter(
                        x=grah['Date'],
                        y=grah['Duty rate in pence/litre (ULSP)'],
                        mode='markers',
                        name='Duty')
else:
    price = go.Scatter(
                        x=grah['Date'],
                        y=grah['Pump price in pence/litre (ULSD)'],
                        mode='markers',
                        name='Price')
    duty = go.Scatter(
                        x=grah['Date'],
                        y=grah['Duty rate in pence/litre (ULSD)'],
                        mode='markers',
                        name='Duty')
   
makegrah = [price, duty]
layout = go.Layout(title = select,
                    xaxis=dict(title='Year'),
                    yaxis=dict(title='Price'))           
fig = go.Figure(data=makegrah, layout=layout)

if select != 'none':
    st.plotly_chart(fig, use_container_width=True)

#ข้อมูล
st.header("Example data")
num_show = st.sidebar.slider('Slide to display Example data.', 1, len(data), 5)
st.table(data[:num_show])


#contact
from PIL import Image
st.header("Member")
st.write("Click on picture to contact")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<h5 style='text-align: center; color: white;'><a href='https://www.instagram.com/yeahhfah/'><img src='https://profile.line-scdn.net/0hCq9w3nFSHF4UDgjBUxBiIWReHzQ3f0VMPDwDO3EIFj4oOl0MbWAHPnYPFj58NltYOG0Ea3NcRWkYHWs4CljgahM-QmwsPlgJMGBXuQ/preview' style='width:200px;height:200px;'></a></h5>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: white;'>65160034<br>นางสาววรกมล สหัสธารา </h6>", unsafe_allow_html=True)

with col2:
    st.markdown("<h5 style='text-align: center; color: white;'><a href='https://www.instagram.com/nnut_jrk/'><img src='https://scontent.fbkk30-1.fna.fbcdn.net/v/t1.6435-9/113633081_2710960809139935_8819572495850754601_n.jpg?_nc_cat=102&ccb=1-7&_nc_sid=730e14&_nc_eui2=AeEI4iE3vLwA8cwak5Iu-LUx6Qffl0VE-0PpB9-XRUT7Q2G977uGcLFmvv3bG1ib2F_H7t6Hc062hoCoItG9Ax6d&_nc_ohc=HH2D2AM8WEYAX8YWlXk&_nc_ht=scontent.fbkk30-1.fna&oh=00_AfAZM8oy7mbk5FIL6pOaepmwbMi9y-u0WjWUvn2VONwa3g&oe=643465A3' style='width:200px;height:200px;'></a></h5>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: white;'>65160035<br>นายจักรินทร์ พูนจบ</h6>", unsafe_allow_html=True)

with col3:
    st.markdown("<h5 style='text-align: center; color: white;'><a href='https://www.instagram.com/fleshntw/'><img src='https://profile.line-scdn.net/0hritPKKNxLUZZKTgAlq5TOSl5Lix6WHRUc01mImt6Jn5gTWpFdE4wcz4he3ViHD4QcB03ID57cXNVOlogR3_Rcl4Zc3RhGWkRfUdmoQ/preview' style='width:200px;height:200px;'></a></h5>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: white;'>65160048<br>นายณัฐวัฒน์ สุวรรณรินทร์ </h6>", unsafe_allow_html=True)