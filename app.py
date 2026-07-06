import streamlit as tf_st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime
from src.models_inference import EmotionPredictor

# Set up beautiful browser tab layouts
tf_st.set_page_config(page_title="AI Emotion Detection Dashboard", layout="wide")

# Initialize our combined AI predictor core
@tf_st.cache_resource
def load_predictor():
    return EmotionPredictor()

try:
    predictor = load_predictor()
except Exception as e:
    tf_st.error(f"Error loading model assets: {e}")
    tf_st.info("Please verify that your models folder has all your downloaded Kaggle files.")

# Setup local history logs tracking file path
HISTORY_FILE = "history.csv"

def save_to_history(text, top_emotion, confidence, model_type):
    new_data = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Student Text": text,
        "Detected Emotion": top_emotion,
        "Confidence": f"{confidence:.2%}",
        "Model Engine": model_type
    }])
    if not os.path.exists(HISTORY_FILE):
        new_data.to_csv(HISTORY_FILE, index=False)
    else:
        new_data.to_csv(HISTORY_FILE, mode='a', header=False, index=False)

# --- SIDEBAR INTERFACE LAYOUT ---
tf_st.sidebar.title("⚙️ Engine Control Panel")
model_choice = tf_st.sidebar.radio(
    "Select Model Architecture:",
    ("BiLSTM (Fast Neural Network)", "BERT (Advanced Transformer)")
)

tf_st.sidebar.markdown("---")
tf_st.sidebar.info(
    "💡 **Tip:** BERT provides deep context tracking, while BiLSTM delivers instant inference execution loops."
)

# --- MAIN DASHBOARD LAYOUT ---
tf_st.title("🧠 Student Sentiment & Emotion Analytics Engine")
tf_st.subheader("Real-time emotional tracking feedback pipeline for modern classrooms.")

user_input = tf_st.text_area(
    "Enter student feedback, query, or reflection text below:",
    placeholder="Type something here (e.g., 'I am stuck on this coding loop and it is completely frustrating...')"
)

if tf_st.button("Analyze Sentiment Distribution", type="primary"):
    if user_input.strip() == "":
        tf_st.warning("Please enter some text before analyzing!")
    else:
        # Route processing through user's active sidebar layout engine choice
        with tf_st.spinner("Running deep analytical processing loops..."):
            if "BiLSTM" in model_choice:
                prob_distributions = predictor.predict_bilstm(user_input)
                engine_tag = "BiLSTM"
            else:
                prob_distributions = predictor.predict_bert(user_input)
                engine_tag = "BERT"
                
        # Calculate primary emotion classifications
        sorted_emotions = sorted(prob_distributions.items(), key=lambda x: x[1], reverse=True)
        primary_emotion = sorted_emotions[0][0]
        primary_conf = sorted_emotions[0][1]
        
        # Save records automatically into structural analytics database logs
        save_to_history(user_input, primary_emotion, primary_conf, engine_tag)
        
        # Display clean layout KPI alert cards
        col1, col2 = tf_st.columns(2)
        with col1:
            tf_st.metric(label="Primary Sentiment Class Detected", value=primary_emotion)
        with col2:
            tf_st.metric(label="Engine Confidence Level", value=f"{primary_conf:.2%}")
            
        tf_st.markdown("### 📊 Distribution Breakdown Matrix")
        
        # Plotly chart processing
        chart_df = pd.DataFrame({
            'Emotion Category': list(prob_distributions.keys()),
            'Confidence Score': list(prob_distributions.values())
        }).sort_values(by='Confidence Score', ascending=True)
        
        fig = px.bar(
            chart_df, 
            x='Confidence Score', 
            y='Emotion Category', 
            orientation='h',
            color='Confidence Score',
            color_continuous_scale='Blues',
            text_auto='.2%'
        )
        fig.update_layout(showlegend=False, height=350)
        tf_st.plotly_chart(fig, use_container_width=True)

# --- VIEW ANALYTICS LOGS SECTION ---
tf_st.markdown("---")
with tf_st.expander("📂 View Historical Assessment Database Logs"):
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
        tf_st.dataframe(history_df.tail(10), use_container_width=True)
    else:
        tf_st.info("No query logs saved inside database sheets yet. Run your first inference check above!")