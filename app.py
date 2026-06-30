import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------
# Load Models
# ----------------------------------------------------

dt_model = joblib.load("best_model.pkl")

try:
    knn_model = joblib.load("knn_model.pkl")
except FileNotFoundError:
    knn_model = None

# ----------------------------------------------------
# Custom CSS
# ----------------------------------------------------

st.markdown("""
<style>

.stApp{
    background:#0E1117;
}

/* Hide Streamlit Footer */
footer{
    visibility:hidden;
}

header{
    visibility:hidden;
}

/* Rainbow Title */

.rainbow{

font-size:55px;
font-weight:bold;
text-align:center;

background:linear-gradient(
90deg,
#ff0000,
#ff7300,
#fffb00,
#48ff00,
#00ffd5,
#002bff,
#7a00ff,
#ff00ab);

-webkit-background-clip:text;
-webkit-text-fill-color:transparent;

}

/* Subtitle */

.subtitle{

text-align:center;
font-size:24px;
color:white;
margin-bottom:20px;

}

/* Cards */
.card{
    background:#161B22;
    padding:25px;
    border-radius:18px;
    border:1px solid #00E5FF;
    color:white !important;
    box-shadow:0px 0px 15px rgba(0,229,255,0.15);
}

.card h1,
.card h2,
.card h3,
.card h4,
.card h5,
.card h6,
.card p,
.card li{
    color:white !important;
}

.metric{
    background:#1F2937;
    padding:20px;
    border-radius:15px;
    text-align:center;
    border:1px solid #00E5FF;
    color:white !important;
}

.metric h2,
.metric p{
    color:white !important;
}

/* Buttons */

.stButton>button{

width:100%;

height:55px;

background:linear-gradient(90deg,#00E5FF,#0099FF);

color:white;

font-size:20px;

font-weight:bold;

border:none;

border-radius:12px;

}

.stButton>button:hover{

background:linear-gradient(90deg,#00C2FF,#007BFF);

}

/* Sidebar */

section[data-testid="stSidebar"]{

background:#111827;

}


/* Page Text */

h1,h2,h3,h4,h5,h6{
    color:white;
}

p{
    color:white;
}

label{
    color:white;
}


/* Markdown */

.stMarkdown{
    color:white !important;
}

/* Labels */

.stSelectbox label,
.stNumberInput label{
    color:white !important;
    font-weight:600;
}

/* Caption */

[data-testid="stCaptionContainer"]{
    color:#D1D5DB !important;
}
            
/* Closed Select Box */
.stSelectbox div[data-baseweb="select"]{
    color:white !important;
}
     
            
 /* Metric Card */

[data-testid="stMetric"]{
    background:#1F2937;
    padding:18px;
    border-radius:15px;
    border:1px solid #00E5FF;
}

/* Metric */

[data-testid="stMetricValue"]{
    color:#00E5FF !important;
    font-size:34px;
    font-weight:bold;
}

[data-testid="stMetricLabel"]{
    color:white !important;
    font-size:18px;
}

</style>

""",unsafe_allow_html=True)

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.title("💼 Employee Salary Prediction")

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Project")
st.sidebar.write("Employee Salary Prediction using Machine Learning.")

st.sidebar.markdown("---")

st.sidebar.subheader("🤖 Models")
st.sidebar.success("Decision Tree Regressor")

if knn_model is not None:
    st.sidebar.success("KNN Regressor")

st.sidebar.markdown("---")

st.sidebar.subheader("📈 Best Model")
st.sidebar.success("🌳 Decision Tree")
st.sidebar.write("R² Score : **99.11%**")

st.sidebar.markdown("---")

st.sidebar.subheader("🛠 Technologies")

st.sidebar.write("""
- Python
- Streamlit
- Pandas
- Scikit-Learn
- Joblib
""")

st.sidebar.markdown("---")

st.sidebar.write("👨‍💻 Developed by")
st.sidebar.success("Khaja Mainuddin")


# ----------------------------------------------------
# Hero Section
# ----------------------------------------------------

st.markdown("""
<div class='rainbow'>
💼 Employee Salary Prediction
</div>

<div class='subtitle'>
AI Powered Human Resource Salary Estimation System
</div>
""",unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------
# Dashboard Cards
# ----------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "99.11%")
col2.metric("Models", "2")
col3.metric("Optimization", "GridSearchCV")
col4.metric("Task", "Regression")

st.divider()

# ----------------------------------------------------
# Model Information
# ----------------------------------------------------

st.markdown("## 📈 Model Performance")

model1, model2 = st.columns(2)

with model1:

    st.success("""
🌳 **Decision Tree Regressor**

R² Score : **99.11%**

✔ Best Performing Model
""")

with model2:

    st.info("""
🤝 **KNN Regressor**

R² Score : **98.73%**

✔ Excellent Performance
""")

st.divider()

# ----------------------------------------------------
# Prediction Section Heading
# ----------------------------------------------------

st.markdown(
    "<h2 style='color:white;'>📝 Employee Information</h2>",
    unsafe_allow_html=True
)

st.caption(
    "Enter the employee details below and click **Predict Salary** to estimate the annual salary."
)
st.write("")

# -----------------------------
# Model Selection
# -----------------------------

if knn_model is not None:
    selected_model = st.selectbox(
        "🤖 Select Machine Learning Model",
        ["Decision Tree Regressor", "KNN Regressor"]
    )
else:
    selected_model = "Decision Tree Regressor"
    st.info("Using Decision Tree Regressor")

st.write("")

# -----------------------------
# Input Form
# -----------------------------

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=65,
        value=25
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    department = st.selectbox(
        "Department",
        [
            "Engineering",
            "Sales",
            "Finance",
            "HR",
            "Marketing",
            "Product"
        ]
    )

    job_title = st.selectbox(
        "Job Title",
        [
            "Engineer",
            "Executive",
            "Intern",
            "Analyst",
            "Manager"
        ]
    )

with col2:

    experience = st.number_input(
        "Experience (Years)",
        min_value=0,
        max_value=40,
        value=2
    )

    education = st.selectbox(
        "Education Level",
        [
            "Bachelor",
            "Master",
            "PhD"
        ]
    )

    location = st.selectbox(
        "Location",
        [
            "Austin",
            "Seattle",
            "New York",
            "San Francisco",
            "Chicago"
        ]
    )

st.write("")

# ====================================================
# Prediction Button
# ====================================================

st.markdown("""
<div style="display:flex; justify-content:center; margin-top:20px; margin-bottom:20px;">
</div>
""", unsafe_allow_html=True)

left, center, right = st.columns([4, 1, 4])

with center:
    predict = st.button("💰 Estimate Salary", use_container_width=True)

# ====================================================
# Prediction
# ====================================================

if predict:

    # Create Input DataFrame
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Department": [department],
        "Job_Title": [job_title],
        "Experience_Years": [experience],
        "Education_Level": [education],
        "Location": [location]
    })

    # Select Model
    if selected_model == "Decision Tree Regressor":
        prediction = dt_model.predict(input_data)
        salary = prediction[0]
        r2_score = "99.11%"
        model_used = "🌳 Decision Tree Regressor"

    else:
        prediction = knn_model.predict(input_data)
        salary = prediction[0]
        r2_score = "98.73%"
        model_used = "🤝 KNN Regressor"

    st.write("")

    # ====================================================
    # Prediction Result Card
    # ====================================================

    st.markdown(f"""
<div style="
    background:linear-gradient(135deg,#00E5FF,#007BFF);
    padding:35px;
    border-radius:20px;
    text-align:center;
    color:white;
    box-shadow:0px 0px 25px rgba(0,229,255,0.35);
">

<h2>💰 Predicted Employee Salary</h2>

<p style="font-size:18px; margin-bottom:10px;">
Estimated Annual Salary
</p>

<h1 style="font-size:60px; font-weight:bold;">
₹ {salary:,.2f}
</h1>

<hr style="border:1px solid rgba(255,255,255,0.4);">

<h3>{model_used}</h3>

<p style="font-size:18px;">
Model Accuracy (R² Score): <b>{r2_score}</b>
</p>

</div>
""", unsafe_allow_html=True) 

    st.markdown("<br>", unsafe_allow_html=True)

    # ====================================================
    # Model Comparison
    # ====================================================

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Selected Model",
            value=selected_model
        )

    with col2:
        st.metric(
            label="R² Score",
            value=r2_score
        )

# ====================================================
# Footer
# ====================================================
st.divider()

st.markdown("""
<div style="
text-align:center;
color:#A0A0A0;
padding:10px 0;
font-size:15px;
">

👨‍💻 <b>Developed by Khaja Mainuddin</b>

</div>
""", unsafe_allow_html=True)




