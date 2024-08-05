import streamlit as st
import pickle5 as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def get_clean_data():
  data = pd.read_csv("C:/Users/Pavithra/streamlit-cancer-predict-main/data/data.csv")
  
  #data = data.drop(['Unnamed: 32', 'id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data     


def add_sidebar():
  st.sidebar.header("Cell Nuclei Measurements")
  
  data = get_clean_data()
  
  slider_labels = [
        ("Radius (Red blood cells)", "radius_mean"),
        ("Texture (Red blood cells)", "texture_mean"),
        ("Perimeter (Red blood cells)", "perimeter_mean"),
        ("Area (Red blood cells)", "area_mean"),
        ("Smoothness (Red blood cells)", "smoothness_mean"),
        ("Compactness (Red blood cells)", "compactness_mean"),
        ("Concavity (Red blood cells)", "concavity_mean"),
        ("Concave points (Red blood cells)", "concave points_mean"),
        ("Symmetry (Red blood cells)", "symmetry_mean"),
        ("Fractal dimension (Red blood cells)", "fractal_dimension_mean"),
        ("Radius (White blood cells)", "radius_se"),
        ("Texture (White blood cells)", "texture_se"),
        ("Perimeter (White blood cells)", "perimeter_se"),
        ("Area (White blood cells)", "area_se"),
        ("Smoothness (White blood cells)", "smoothness_se"),
        ("Compactness (White blood cells)", "compactness_se"),
        ("Concavity (White blood cells)", "concavity_se"),
        ("Concave points (White blood cells)", "concave points_se"),
        ("Symmetry (White blood cells)", "symmetry_se"),
        ("Fractal dimension (White blood cells)", "fractal_dimension_se"),
        ("Radius (platelets)", "radius_worst"),
        ("Texture (platelets)", "texture_worst"),
        ("Perimeter (platelets)", "perimeter_worst"),
        ("Area (platelets)", "area_worst"),
        ("Smoothness (platelets)", "smoothness_worst"),
        ("Compactness (platelets)", "compactness_worst"),
        ("Concavity (platelets)", "concavity_worst"),
        ("Concave points (platelets)", "concave points_worst"),
        ("Symmetry (platelets)", "symmetry_worst"),
        ("Fractal dimension (platelets)", "fractal_dimension_worst"),
    ]

  input_dict = {}

  for label, key in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean())
    )
    
  return input_dict


def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data.drop(['diagnosis'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict
  

def get_radar_chart(input_data):
  
  input_data = get_scaled_values(input_data)
  
  categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Red blood cells Value'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='White blood cells value'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='platelets Value'
  ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
  return fig


def add_predictions(input_data):
  model = pickle.load(open("C:/Users/Pavithra/streamlit-cancer-predict-main/model/model.pkl", "rb"))
  scaler = pickle.load(open("C:/Users/Pavithra/streamlit-cancer-predict-main/model/scaler.pkl", "rb"))
  
  input_array = np.array(list(input_data.values())).reshape(1, -1)
  
  input_array_scaled = scaler.transform(input_array)
  
  prediction = model.predict(input_array_scaled)
  
  st.subheader("Cell cluster prediction")
  st.write("The cell cluster is:")
  
  if prediction[0] == 0:
    st.write("<span class='diagnosis benign'>It is Lekumia Cancer</span>", unsafe_allow_html=True)
  else:
    st.write("<span class='diagnosis malicious'>Not a Cancer</span>", unsafe_allow_html=True)
    
  
  st.write("Probability of being Lekumia Cancer: ", model.predict_proba(input_array_scaled)[0][0])
  st.write("Probability of being Not a Cancer: ", model.predict_proba(input_array_scaled)[0][1])
  
  st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")



def main():
  st.set_page_config(    page_title="Lekumia Cancer Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )
  
  with open("C:/Users/Pavithra/streamlit-cancer-predict-main/assets/style.css") as f:
     st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
  

  input_data =add_sidebar()
  
  with st.container():
    st.title("Lekumia Cancer Predictor")    
    st.write("Please connect this app to lekumia blood cancer . This app predicts using a machine learning model whether a lekumia cancer or Not based on the measurements it receives from your Dataset. ")
  
  
  col1, col2= st.columns([4,1])
  
  
  with col1:
    radar_chart=get_radar_chart(input_data)
    st.plotly_chart(radar_chart)
  with col2:
    add_predictions(input_data)


 
if __name__ == '__main__':
  main()