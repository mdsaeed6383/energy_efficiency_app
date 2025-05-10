import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

model= load_model('my_best_pipeline')


st.title('energy Efficiency Predictor')
st.markdown('predict the **Heating Load (Y1)** of a building based on its design features.')

st.sidebar.header('Input Features')

relative_compactness= st.sidebar.slider('Relative Compactness',0.5,1.0,0.8)
surface_area=st.sidebar.slider('Surface Area', 500, 900, 650)
wall_area=st.sidebar.slider('Wall Area', 200, 500, 300)
roof_area=st.sidebar.slider('Roof Area', 100,300, 150)
overall_height=st.sidebar.selectbox('Overall Height',[3.5,7.0])
orientation=st.sidebar.selectbox('Orientation',[2,3,4,5])
glazing_area=st.sidebar.slider("Glazing Area", 0.0, 0.4, 0.1)
glazing_area_dist=st.sidebar.selectbox('Glazing Area Distribution',[0, 1, 3, 4, 5])
cooling_load=st.sidebar.slider('Cooling', 0.0,60.0,25.0)



input_df= pd.DataFrame({
    'Relative Compactness':[relative_compactness],
    'Surface Area':[surface_area],
    'Wall Area':[wall_area],
    'Roof Area':[roof_area],
    'Overall Height':[overall_height],
    'Orientation':[orientation],
    'Glazing Area':[glazing_area],
    'Glazing Area Distribution':[glazing_area_dist],
    'Cooling Load':[cooling_load]


})

if st.button('Predict Heating Load'):
    Prediction=predict_model(model, data=input_df)
    st.subheader("Predicted Heating Load:")
    st.success(f"{Prediction['prediction_label'][0]:.2f} kWh/m^2")
    