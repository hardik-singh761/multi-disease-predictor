from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load models
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('layout.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    prediction = None
    input_data = {}
    if request.method == 'POST':
        fields = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
        input_data = {field: request.form.get(field) for field in fields}
        values = np.asarray([float(input_data[f]) for f in fields]).reshape(1, -1)
        result = diabetes_model.predict(values)[0]
        prediction = "Diabetic" if result == 1 else "Not Diabetic"
    return render_template('diabetes.html', prediction=prediction, data=input_data)

@app.route('/heart', methods=['GET', 'POST'])
def heart():
    prediction = None
    sex_options = ['Male', 'Female']
    cp_options = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
    fbs_options = ['> 120 mg/dl', '<= 120 mg/dl']
    restecg_options = ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy']
    exang_options = ['Yes', 'No']
    slope_options = ['Upsloping', 'Flat', 'Downsloping']
    thal_options = ['Normal', 'Fixed Defect', 'Reversible Defect']

    form_data = request.form
    if request.method == 'POST':
        try:
            sex_map = {'Male': 1, 'Female': 0}
            cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
            fbs_map = {'> 120 mg/dl': 1, '<= 120 mg/dl': 0}
            restecg_map = {'Normal': 0, 'ST-T Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
            exang_map = {'Yes': 1, 'No': 0}
            slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
            thal_map = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}

            input_data = [
                float(form_data['age']),
                sex_map[form_data['sex']],
                cp_map[form_data['cp']],
                float(form_data['trestbps']),
                float(form_data['chol']),
                fbs_map[form_data['fbs']],
                restecg_map[form_data['restecg']],
                float(form_data['thalach']),
                exang_map[form_data['exang']],
                float(form_data['oldpeak']),
                slope_map[form_data['slope']],
                float(form_data['ca']),
                thal_map[form_data['thal']]
            ]

            input_np = np.asarray(input_data).reshape(1, -1)
            result = heart_model.predict(input_np)[0]
            prediction = "Heart Disease Detected" if result == 1 else "No Heart Disease"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('heart.html', prediction=prediction, data=form_data,
                           sex_options=sex_options,
                           cp_options=cp_options,
                           fbs_options=fbs_options,
                           restecg_options=restecg_options,
                           exang_options=exang_options,
                           slope_options=slope_options,
                           thal_options=thal_options)


@app.route('/parkinsons', methods=['GET', 'POST'])
def parkinsons():
    prediction = None
    input_data = {}
    if request.method == 'POST':
        fields = ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)',
                  'MDVP:RAP','MDVP:PPQ','Jitter:DDP','MDVP:Shimmer','MDVP:Shimmer(dB)',
                  'Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA','NHR','HNR',
                  'RPDE','DFA','spread1','spread2','D2','PPE']
        input_data = {field: request.form.get(field) for field in fields}
        values = np.asarray([float(input_data[f]) for f in fields]).reshape(1, -1)
        result = parkinsons_model.predict(values)[0]
        prediction = "Parkinson’s Detected" if result == 1 else "No Parkinson’s"
    return render_template('parkinsons.html', prediction=prediction, data=input_data)

if __name__ == '__main__':
    app.run(debug=True)
