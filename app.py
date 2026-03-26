import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
import pickle
from PIL import Image, ImageOps

import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import xgboost as xgb

# import datacodes as datac

# Load trained model
classifier1 = pickle.load(open('SVC_Trainmodel2.pkl', 'rb'))
classifier2 = pickle.load(open('xgb_Trainmodel1.pkl', 'rb'))


Genes_in_mother_side_Box = ['No', 'Yes']
Num_Genes_in_mother_side = [0, 1]
Genes_in_mother_side_Len = len(Num_Genes_in_mother_side)

Inherited_from_father_Box = ['No', 'Yes']
Num_Inherited_from_father = [0, 1]
Inherited_from_father_Len = len(Num_Inherited_from_father)

Maternal_gene_Box = ['No', 'Yes']
Num_Maternal_gene = [0, 1]
Maternal_gene_Len = len(Num_Maternal_gene)

Paternal_gene_Box = ['No', 'Yes']
Num_Paternal_gene = [0, 1]
Paternal_gene_Len = len(Num_Paternal_gene)

Status_Box = ['Deceased', 'Alive']
Num_Status = [0, 1]
Status_Len = len(Num_Status)

Respiratory_Rate_Box = ['Tachypnea', 'Normal (30-60)']
Num_Respiratory_Rate = [0, 1]
Respiratory_Rate_Len = len(Num_Respiratory_Rate)

Heart_Rate_Box = ['Tachycardia', 'Normal']
Num_Heart_Rate = [0, 1]
Heart_Rate_Len = len(Num_Heart_Rate)

Test_1_Box = ['No', 'Yes']
Num_Test_1 = [0, 1]
Test_1_Len = len(Num_Test_1)

Test_2_Box = ['No', 'Yes']
Num_Test_2 = [0, 1]
Test_2_Len = len(Num_Test_2)

Test_3_Box = ['No', 'Yes']
Num_Test_3 = [0, 1]
Test_3_Len = len(Num_Test_3)

Test_4_Box = ['No', 'Yes']
Num_Test_4 = [0, 1]
Test_4_Len = len(Num_Test_4)

Test_5_Box = ['No', 'Yes']
Num_Test_5 = [0, 1]
Test_5_Len = len(Num_Test_5)

Parental_consent_Box = ['No', 'Yes']
Num_Parental_consent = [0, 1]
Parental_consent_Len = len(Num_Parental_consent)

Follow_up_Box = ['Low', 'High']
Num_Follow_up = [0, 1]
Follow_up_Len = len(Num_Follow_up)

Gender_Box = ['Female', 'Male', 'Ambiguous']
Num_Gender = [0, 1, 2]
Gender_Len = len(Num_Gender)

Birth_asphyxia_Box = ['No', 'Yes']
Num_Birth_asphyxia = [0, 1]
Birth_asphyxia_Len = len(Num_Birth_asphyxia)

Autopsy_shows_birth_defect_Box = ['No', 'Yes', 'None', 'Not applicable']
Num_Autopsy_shows_birth_defect = [0, 1, 2, 3]
Autopsy_shows_birth_defect_Len = len(Num_Autopsy_shows_birth_defect)

Folic_acid_details_Box = ['No', 'Yes']
Num_Folic_acid_details = [0, 1]
Folic_acid_details_Len = len(Num_Folic_acid_details)

H_O_serious_maternal_illness_Box = ['No', 'Yes']
Num_H_O_serious_maternal_illness = [0, 1]
H_O_serious_maternal_illness_Len = len(Num_H_O_serious_maternal_illness)

H_O_radiation_exposure_Box = ['No', 'Yes']
Num_H_O_radiation_exposure = [0, 1]
H_O_radiation_exposure_Len = len(Num_H_O_radiation_exposure)

H_O_substance_abuse_Box = ['No', 'Yes']
Num_H_O_substance_abuse = [0, 1]
H_O_substance_abuse_Len = len(Num_H_O_substance_abuse)

Assisted_conception_Box = ['No', 'Yes']
Num_Assisted_conception = [0, 1]
Assisted_conception_Len = len(Num_Assisted_conception)

History_Box = ['No', 'Yes']
Num_History = [0, 1]
History_Len = len(Num_History)

Birth_defects_Box = ['Multiple', 'Singular']
Num_Birth_defects = [0, 1]
Birth_defects_Len = len(Num_Birth_defects)

Blood_test_result_Box = ['normal', 'slightly abnormal', 'inconclusive', 'abnormal']
Num_Blood_test_result = [0, 1, 2, 3]
Blood_test_result_Len = len(Num_Blood_test_result)


Disorder_Subclass_Box = ["Cystic fibrosis", "Leber's hereditary optic neuropathy", "Diabetes", "Leigh syndrome", "Cancer", "Tay-Sachs", "Hemochromatosis", "Mitochondrial myopathy", "Alzheimer's"] 
Num_Disorder_Subclass = [0, 1, 2, 3, 4, 5, 6, 7, 8]
Disorder_Subclass_Len = len(Num_Disorder_Subclass)



def prediction1(Patient_Age, Genes_in_mother_side, Inherited_from_father, Maternal_gene, Paternal_gene, 
               Blood_cell_count, Mother_age, Father_age, Status, Respiratory_Rate, Heart_Rate, Test_1, 
               Test_2, Test_3, Test_4, Test_5, Parental_consent, Follow_up, Gender, Birth_asphyxia, 
               Autopsy_shows_birth_defect, Folic_acid_details, H_O_serious_maternal_illness, 
               H_O_radiation_exposure, H_O_substance_abuse, Assisted_conception, History_of_anomalies_in_prev_pregnancies, 
               No_of_previous_abortion, Birth_defects, White_Blood_cell_count, Blood_test_result, 
               Symptom_1, Symptom_2, Symptom_3, Symptom_4, Symptom_5
               , sum_of_Mom_dad_age_avg, total_symptom):

    for i in range(Genes_in_mother_side_Len):
        if Genes_in_mother_side == Genes_in_mother_side_Box[i]:
            Genes_in_mother_side = Num_Genes_in_mother_side[i]

    for i in range(Inherited_from_father_Len):
        if Inherited_from_father == Inherited_from_father_Box[i]:
            Inherited_from_father = Num_Inherited_from_father[i]

    for i in range(Maternal_gene_Len):
        if Maternal_gene == Maternal_gene_Box[i]:
            Maternal_gene = Num_Maternal_gene[i]

    for i in range(Paternal_gene_Len):
        if Paternal_gene == Paternal_gene_Box[i]:
            Paternal_gene = Num_Paternal_gene[i]
            
    for i in range(Status_Len):
        if Status == Status_Box[i]:
            Status = Num_Status[i]

    for i in range(Respiratory_Rate_Len):
        if Respiratory_Rate == Respiratory_Rate_Box[i]:
            Respiratory_Rate = Num_Respiratory_Rate[i]


    for i in range(Heart_Rate_Len):
        if Heart_Rate == Heart_Rate_Box[i]:
            Heart_Rate = Num_Heart_Rate[i]

    for i in range(Test_1_Len):
        if Test_1 == Test_1_Box[i]:
            Test_1 = Num_Test_1[i]
            
    for i in range(Test_2_Len):
        if Test_2 == Test_2_Box[i]:
            Test_2 = Num_Test_2[i]

    for i in range(Test_3_Len):
        if Test_3 == Test_3_Box[i]:
            Test_3 = Num_Test_3[i]


    for i in range(Test_4_Len):
        if Test_4 == Test_4_Box[i]:
            Test_4 = Num_Test_4[i]

    for i in range(Test_5_Len):
        if Test_5 == Test_5_Box[i]:
            Test_5 = Num_Test_5[i]
            
    for i in range(Parental_consent_Len):
        if Parental_consent == Parental_consent_Box[i]:
            Parental_consent = Num_Parental_consent[i]

    for i in range(Follow_up_Len):
        if Follow_up == Follow_up_Box[i]:
            Follow_up = Num_Follow_up[i]
            
           
    for i in range(Gender_Len):
        if Gender == Gender_Box[i]:
            Gender = Num_Gender[i]

    for i in range(Birth_asphyxia_Len):
        if Birth_asphyxia == Birth_asphyxia_Box[i]:
            Birth_asphyxia = Num_Birth_asphyxia[i]
            
    for i in range(Autopsy_shows_birth_defect_Len):
        if Autopsy_shows_birth_defect == Autopsy_shows_birth_defect_Box[i]:
            Autopsy_shows_birth_defect = Num_Autopsy_shows_birth_defect[i]

    for i in range(Folic_acid_details_Len):
        if Folic_acid_details == Folic_acid_details_Box[i]:
            Folic_acid_details = Num_Folic_acid_details[i]


    for i in range(H_O_serious_maternal_illness_Len):
        if H_O_serious_maternal_illness == H_O_serious_maternal_illness_Box[i]:
            H_O_serious_maternal_illness = Num_H_O_serious_maternal_illness[i]

    for i in range(H_O_radiation_exposure_Len):
        if H_O_radiation_exposure == H_O_radiation_exposure_Box[i]:
            H_O_radiation_exposure = Num_H_O_radiation_exposure[i]
            
    for i in range(H_O_substance_abuse_Len):
        if H_O_substance_abuse == H_O_substance_abuse_Box[i]:
            H_O_substance_abuse = Num_H_O_substance_abuse[i]

    for i in range(Assisted_conception_Len):
        if Assisted_conception == Assisted_conception_Box[i]:
            Assisted_conception = Num_Assisted_conception[i]


    for i in range(History_Len):
        if History_of_anomalies_in_prev_pregnancies == History_Box[i]:
            History_of_anomalies_in_prev_pregnancies = Num_History[i]

    for i in range(Birth_defects_Len):
        if Birth_defects == Birth_defects_Box[i]:
            Birth_defects = Num_Birth_defects[i]
            
    for i in range(Blood_test_result_Len):
        if Blood_test_result == Blood_test_result_Box[i]:
            Blood_test_result = Num_Blood_test_result[i]
                
    values = np.array([[Patient_Age, Genes_in_mother_side, Inherited_from_father, Maternal_gene, 
                        Paternal_gene, Blood_cell_count, Mother_age, Father_age, Status, Respiratory_Rate, 
                        Heart_Rate, Test_1, Test_2, Test_3, Test_4, Test_5, Parental_consent, Follow_up, 
                        Gender, Birth_asphyxia, Autopsy_shows_birth_defect, Folic_acid_details, 
                        H_O_serious_maternal_illness, H_O_radiation_exposure, H_O_substance_abuse, 
                        Assisted_conception, History_of_anomalies_in_prev_pregnancies, No_of_previous_abortion, 
                        Birth_defects, White_Blood_cell_count, Blood_test_result, Symptom_1, Symptom_2, 
                        Symptom_3, Symptom_4, Symptom_5
                        , sum_of_Mom_dad_age_avg, total_symptom]])
    
    predicted = classifier1.predict(values)
                                
    print("final_prediction:", predicted)
    
    if predicted == 0:
        pred = "Cystic fibrosis"
    elif predicted == 1:
        pred = "Leber's hereditary optic neuropathy"
    elif predicted == 2:
        pred = "Diabetes"
    elif predicted == 3:
        pred = "Leigh syndrome"
    elif predicted == 4:
        pred = "Cancer"
    elif predicted == 5:
        pred = "Tay-Sachs"
    elif predicted == 6:
        pred = "Hemochromatosis"
    elif predicted == 7:
        pred = "Mitochondrial myopathy"
    elif predicted == 8:
        pred = "Alzheimer's"
    return pred
################predection 2 ####################################

def prediction2(Patient_Age, Genes_in_mother_side, Inherited_from_father, Maternal_gene, Paternal_gene, 
               Blood_cell_count, Mother_age, Father_age, Status, Respiratory_Rate, Heart_Rate, Test_1, 
               Test_2, Test_3, Test_4, Test_5, Parental_consent, Follow_up, Gender, Birth_asphyxia, 
               Autopsy_shows_birth_defect, Folic_acid_details, H_O_serious_maternal_illness, 
               H_O_radiation_exposure, H_O_substance_abuse, Assisted_conception, History_of_anomalies_in_prev_pregnancies, 
               No_of_previous_abortion, Birth_defects, White_Blood_cell_count, Blood_test_result, 
               Symptom_1, Symptom_2, Symptom_3, Symptom_4, Symptom_5, Disorder_Subclass
               , sum_of_Mom_dad_age_avg, total_symptom):

    for i in range(Genes_in_mother_side_Len):
        if Genes_in_mother_side == Genes_in_mother_side_Box[i]:
            Genes_in_mother_side = Num_Genes_in_mother_side[i]

    for i in range(Inherited_from_father_Len):
        if Inherited_from_father == Inherited_from_father_Box[i]:
            Inherited_from_father = Num_Inherited_from_father[i]

    for i in range(Maternal_gene_Len):
        if Maternal_gene == Maternal_gene_Box[i]:
            Maternal_gene = Num_Maternal_gene[i]

    for i in range(Paternal_gene_Len):
        if Paternal_gene == Paternal_gene_Box[i]:
            Paternal_gene = Num_Paternal_gene[i]
            
    for i in range(Status_Len):
        if Status == Status_Box[i]:
            Status = Num_Status[i]

    for i in range(Respiratory_Rate_Len):
        if Respiratory_Rate == Respiratory_Rate_Box[i]:
            Respiratory_Rate = Num_Respiratory_Rate[i]


    for i in range(Heart_Rate_Len):
        if Heart_Rate == Heart_Rate_Box[i]:
            Heart_Rate = Num_Heart_Rate[i]

    for i in range(Test_1_Len):
        if Test_1 == Test_1_Box[i]:
            Test_1 = Num_Test_1[i]
            
    for i in range(Test_2_Len):
        if Test_2 == Test_2_Box[i]:
            Test_2 = Num_Test_2[i]

    for i in range(Test_3_Len):
        if Test_3 == Test_3_Box[i]:
            Test_3 = Num_Test_3[i]


    for i in range(Test_4_Len):
        if Test_4 == Test_4_Box[i]:
            Test_4 = Num_Test_4[i]

    for i in range(Test_5_Len):
        if Test_5 == Test_5_Box[i]:
            Test_5 = Num_Test_5[i]
            
    for i in range(Parental_consent_Len):
        if Parental_consent == Parental_consent_Box[i]:
            Parental_consent = Num_Parental_consent[i]

    for i in range(Follow_up_Len):
        if Follow_up == Follow_up_Box[i]:
            Follow_up = Num_Follow_up[i]
            
           
    for i in range(Gender_Len):
        if Gender == Gender_Box[i]:
            Gender = Num_Gender[i]

    for i in range(Birth_asphyxia_Len):
        if Birth_asphyxia == Birth_asphyxia_Box[i]:
            Birth_asphyxia = Num_Birth_asphyxia[i]
            
    for i in range(Autopsy_shows_birth_defect_Len):
        if Autopsy_shows_birth_defect == Autopsy_shows_birth_defect_Box[i]:
            Autopsy_shows_birth_defect = Num_Autopsy_shows_birth_defect[i]

    for i in range(Folic_acid_details_Len):
        if Folic_acid_details == Folic_acid_details_Box[i]:
            Folic_acid_details = Num_Folic_acid_details[i]


    for i in range(H_O_serious_maternal_illness_Len):
        if H_O_serious_maternal_illness == H_O_serious_maternal_illness_Box[i]:
            H_O_serious_maternal_illness = Num_H_O_serious_maternal_illness[i]

    for i in range(H_O_radiation_exposure_Len):
        if H_O_radiation_exposure == H_O_radiation_exposure_Box[i]:
            H_O_radiation_exposure = Num_H_O_radiation_exposure[i]
            
    for i in range(H_O_substance_abuse_Len):
        if H_O_substance_abuse == H_O_substance_abuse_Box[i]:
            H_O_substance_abuse = Num_H_O_substance_abuse[i]

    for i in range(Assisted_conception_Len):
        if Assisted_conception == Assisted_conception_Box[i]:
            Assisted_conception = Num_Assisted_conception[i]


    for i in range(History_Len):
        if History_of_anomalies_in_prev_pregnancies == History_Box[i]:
            History_of_anomalies_in_prev_pregnancies = Num_History[i]

    for i in range(Birth_defects_Len):
        if Birth_defects == Birth_defects_Box[i]:
            Birth_defects = Num_Birth_defects[i]
            
    for i in range(Blood_test_result_Len):
        if Blood_test_result == Blood_test_result_Box[i]:
            Blood_test_result = Num_Blood_test_result[i]

    for i in range(Disorder_Subclass_Len):
        if Disorder_Subclass == Disorder_Subclass_Box[i]:
            Disorder_Subclass = Num_Disorder_Subclass[i]
                
    values = np.array([[Patient_Age, Genes_in_mother_side, Inherited_from_father, Maternal_gene, 
                        Paternal_gene, Blood_cell_count, Mother_age, Father_age, Status, Respiratory_Rate, 
                        Heart_Rate, Test_1, Test_2, Test_3, Test_4, Test_5, Parental_consent, Follow_up, 
                        Gender, Birth_asphyxia, Autopsy_shows_birth_defect, Folic_acid_details, 
                        H_O_serious_maternal_illness, H_O_radiation_exposure, H_O_substance_abuse, 
                        Assisted_conception, History_of_anomalies_in_prev_pregnancies, No_of_previous_abortion, 
                        Birth_defects, White_Blood_cell_count, Blood_test_result, Symptom_1, Symptom_2, 
                        Symptom_3, Symptom_4, Symptom_5, Disorder_Subclass
                        , sum_of_Mom_dad_age_avg, total_symptom]])
    
    predicted = classifier2.predict(values)
                                
    print("final_prediction:", predicted)
    
    if predicted == 0:
        pred = "Multifactorial genetic inheritance disorders"
    elif predicted == 1:
        pred = "Mitochondrial genetic inheritance disorders"
    else:
        pred = "Single-gene inheritance diseases"
    return pred


with st.sidebar:
    choose = option_menu("Genome Disorder Detection", ["Home", "Prediction1", "Prediction2", "Performance"],
                         icons=['house', 'laptop', 'laptop body fill', 'kanban', 'book','person lines fill'],
                         menu_icon="Detection-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

# Main function:
def main():
    if choose == "Home":
        col1, col2 = st.columns( [0.8, 0.2])
        with col1:               # To display the header text using css style
            st.markdown(""" <style> .font {
                font-size:35px ; font-family: 'Cooper Black'; color: Black;} 
                </style> """, unsafe_allow_html=True)
            st.markdown('<p class="font">Advance Genome Disorder Prediction Model Empowered With Machine Learning</p>', unsafe_allow_html=True)    
    
    
        #st.write("ABSTRACT: -A major and essential issue in biomedical research is to predict genome disorder. Genome disorders cause multivariate diseases like cancer, dementia, diabetes, cystic fibrosis, leigh syndrome, etc. which are causes of high mortality rates around the world. In past, theoretical and explanatory-based approaches were introduced to predict genome disorder. With the development of technology, genetic data were improved to cover almost genome and protein then machine and deep learning-based approaches were introduced to predict genome disorder. Parallel machine and deep learning approaches were introduced. In past, many types of research were conducted on genome disorder prediction using supervised, unsupervised, and semi-supervised learning techniques, most of the approaches using binary problem prediction using genetic sequence data. The prediction results of these approaches were uncertain because of their lower accuracy rate and binary class prediction techniques using genome sequence data but not genome disorder patients’ data with his/her history. Most of the techniques used Ribonucleic acid (RNA) gene sequence and were not often capable of handling bid data effectively. Consequently, in this study, the Alex Net as an effective convolutional neural network architecture proposed to develop an advance genome disorder prediction model (AGDPM) for predicting genome multi classes disorder using a large amount of data. AGDPM tested and compare with the pre-trained Alex Net neural network model and AGDPM gives the best results with 89.89% & 81.25% accuracy of training and testing respectively. So, the advance genome disorder prediction model shows the ability to efficiently predict genome disorder and can process a large amount of patients’ genome disorder data with a multi-class prediction method. AGDPM has proved that it is capable to predict single gene inheritance disorder, mitochondrial gene inheritance disorder, and multifactorial gene inheritance disorder with respect to various statistical performance parameters. So, with the help of AGDPM biomedical research will be improved in terms to predict genetic disorders and put control on high mortality rates. ")    
        plogo = Image.open(r'static/images/home.jpg')
        st.image(plogo, width=700 )
        
    elif choose == "Prediction1":
        # Front end Elements:
        html_temp = """
        <div style ="background-color:blue;padding:13px">
        <h1 style ="color:white;text-align:center;">Genome Disorder Detection</h1>
        </div>
        """
        # Display aspect of front end
        st.markdown(html_temp, unsafe_allow_html=True)
        Patient_Age = st.number_input("Patient Age")
        Genes_in_mother_side = st.selectbox("Genes in mother's side", ('No', 'Yes'))
        Inherited_from_father = st.selectbox("Inherited from father", ('No', 'Yes'))
        Maternal_gene = st.selectbox("Maternal gene", ('No', 'Yes'))
        Paternal_gene = st.selectbox("Paternal gene", ('No', 'Yes'))
        Blood_cell_count = st.number_input("Blood cell count (mcL)")
        Mother_age = st.number_input("Mother's age")
        Father_age = st.number_input("Father's age")
        Status = st.selectbox("Status", ('Deceased', 'Alive'))
        Respiratory_Rate = st.selectbox("Respiratory Rate (breaths/min)",  ('Tachypnea', 'Normal (30-60)'))
        st.markdown("""
                <div>
                <h4 style ="color:green;text-align:left;"></h4>
                </div>"""
                , unsafe_allow_html=True)
    
        Heart_Rate = st.selectbox("Heart Rate (rates/min", ('Tachycardia', 'Normal'))
        Test_1 = st.selectbox("Test 1", ('No', 'Yes'))
        Test_2 = st.selectbox("Test 2", ('No', 'Yes'))
        Test_3 = st.selectbox("Test 3", ('No', 'Yes'))    
        Test_4 = st.selectbox("Test 4", ('No', 'Yes'))    
        Test_5 = st.selectbox("Test 5", ('No', 'Yes'))    
        Parental_consent = st.selectbox("Parental consent", ('No', 'Yes'))    
        Follow_up = st.selectbox("Follow-up", ('Low', 'High'))
        Gender = st.selectbox("Gender", ('Female', 'Male', 'Ambiguous'))
        Birth_asphyxia = st.selectbox("Birth asphyxia", ('No', 'Yes'))
        st.markdown("""
                    <div>
                    <h4 style ="color:green;text-align:left;"></h4>
                    </div>"""
                , unsafe_allow_html=True)
    
        Autopsy_shows_birth_defect = st.selectbox("Autopsy shows birth defect (if applicable)", ('No', 'Yes', 'None', 'Not applicable'))
        Folic_acid_details = st.selectbox("Folic acid details (peri-conceptional)", ('No', 'Yes'))
        H_O_serious_maternal_illness = st.selectbox("H/O serious maternal illness", ('No', 'Yes'))
        H_O_radiation_exposure = st.selectbox("H/O radiation exposure (x-ray)", ('No', 'Yes'))
        H_O_substance_abuse = st.selectbox("H/O substance abuse", ('No', 'Yes'))    
        Assisted_conception = st.selectbox("Assisted conception IVF/ART", ('No', 'Yes'))        
        History_of_anomalies_in_prev_pregnancies = st.selectbox("History of anomalies in previous pregnancies", ('No', 'Yes'))    
        No_of_previous_abortion    = st.number_input("No. of previous abortion")
        Birth_defects = st.selectbox("Birth defects", ('Multiple', 'Singular'))    
        White_Blood_cell_count = st.number_input("White Blood cell count (thousand per microliter)")
        st.markdown("""
                    <div>
                    <h4 style ="color:green;text-align:left;"></h4>
                    </div>"""
                , unsafe_allow_html=True)
        Blood_test_result = st.selectbox("Blood test result", ('normal', 'slightly abnormal', 'inconclusive', 'abnormal'))
        Symptom_1 = st.selectbox("Symptom 1", (0, 1))
        Symptom_2 = st.selectbox("Symptom 2", (0, 1))
        Symptom_3 = st.selectbox("Symptom 3", (0, 1))
        Symptom_4 = st.selectbox("Symptom 4", (0, 1))
        Symptom_5 = st.selectbox("Symptom 5", (0, 1))
        Disorder_Subclass = st.selectbox("Disorder Subclass", ("Cystic fibrosis", "Leber's hereditary optic neuropathy", "Diabetes", "Leigh syndrome", "Cancer", "Tay-Sachs", "Hemochromatosis", "Mitochondrial myopathy", "Alzheimer's"))
        sum_of_Mom_dad_age_avg = (Mother_age + Father_age)/2
        #st.markdown(sum_of_Mom_dad_age_avg)
        sum_of_Mom_dad_age_avg = st.number_input("sum_of_Mom_dad_age_avg")
        total_symptom = (Symptom_1 + Symptom_2 + Symptom_3 + Symptom_4 + Symptom_5)/5
        #st.markdown(total_symptom)
        total_symptom = st.number_input("total_symptom")

        # When Predict is clicked:
        if st.button("Predict"):
            result2 = prediction2(Patient_Age, Genes_in_mother_side, Inherited_from_father, Maternal_gene, 
                                 Paternal_gene, Blood_cell_count, Mother_age, Father_age, Status, 
                                 Respiratory_Rate, Heart_Rate, Test_1, Test_2, Test_3, Test_4, Test_5, 
                                 Parental_consent, Follow_up, Gender, Birth_asphyxia, Autopsy_shows_birth_defect, 
                                 Folic_acid_details, H_O_serious_maternal_illness, H_O_radiation_exposure, 
                                 H_O_substance_abuse, Assisted_conception, History_of_anomalies_in_prev_pregnancies, 
                                 No_of_previous_abortion, Birth_defects, White_Blood_cell_count, Blood_test_result, 
                                 Symptom_1, Symptom_2, Symptom_3, Symptom_4, Symptom_5, Disorder_Subclass
                                 , sum_of_Mom_dad_age_avg, total_symptom)
            st.success(f"Your Prediction : {result2}")


    elif choose == "Prediction2":
        # Front end Elements:
        html_temp = """
        <div style ="background-color:blue;padding:13px">
        <h1 style ="color:white;text-align:center;">Disorder Subclass Detection</h1>
        </div>
        """
        # Display aspect of front end
        st.markdown(html_temp, unsafe_allow_html=True)
        Patient_Age = st.number_input("Patient Age")
        Genes_in_mother_side = st.selectbox("Genes in mother's side", ('No', 'Yes'))
        Inherited_from_father = st.selectbox("Inherited from father", ('No', 'Yes'))
        Maternal_gene = st.selectbox("Maternal gene", ('No', 'Yes'))
        Paternal_gene = st.selectbox("Paternal gene", ('No', 'Yes'))
        Blood_cell_count = st.number_input("Blood cell count (mcL)")
        Mother_age = st.number_input("Mother's age")
        Father_age = st.number_input("Father's age")
        Status = st.selectbox("Status", ('Deceased', 'Alive'))
        Respiratory_Rate = st.selectbox("Respiratory Rate (breaths/min)",  ('Tachypnea', 'Normal (30-60)'))
        st.markdown("""
                <div>
                <h4 style ="color:green;text-align:left;"></h4>
                </div>"""
                , unsafe_allow_html=True)
    
        Heart_Rate = st.selectbox("Heart Rate (rates/min", ('Tachycardia', 'Normal'))
        Test_1 = st.selectbox("Test 1", ('No', 'Yes'))
        Test_2 = st.selectbox("Test 2", ('No', 'Yes'))
        Test_3 = st.selectbox("Test 3", ('No', 'Yes'))    
        Test_4 = st.selectbox("Test 4", ('No', 'Yes'))    
        Test_5 = st.selectbox("Test 5", ('No', 'Yes'))    
        Parental_consent = st.selectbox("Parental consent", ('No', 'Yes'))    
        Follow_up = st.selectbox("Follow-up", ('Low', 'High'))
        Gender = st.selectbox("Gender", ('Female', 'Male', 'Ambiguous'))
        Birth_asphyxia = st.selectbox("Birth asphyxia", ('No', 'Yes'))
        st.markdown("""
                    <div>
                    <h4 style ="color:green;text-align:left;"></h4>
                    </div>"""
                , unsafe_allow_html=True)
    
        Autopsy_shows_birth_defect = st.selectbox("Autopsy shows birth defect (if applicable)", ('No', 'Yes', 'None', 'Not applicable'))
        Folic_acid_details = st.selectbox("Folic acid details (peri-conceptional)", ('No', 'Yes'))
        H_O_serious_maternal_illness = st.selectbox("H/O serious maternal illness", ('No', 'Yes'))
        H_O_radiation_exposure = st.selectbox("H/O radiation exposure (x-ray)", ('No', 'Yes'))
        H_O_substance_abuse = st.selectbox("H/O substance abuse", ('No', 'Yes'))    
        Assisted_conception = st.selectbox("Assisted conception IVF/ART", ('No', 'Yes'))        
        History_of_anomalies_in_prev_pregnancies = st.selectbox("History of anomalies in previous pregnancies", ('No', 'Yes'))    
        No_of_previous_abortion    = st.number_input("No. of previous abortion")
        Birth_defects = st.selectbox("Birth defects", ('Multiple', 'Singular'))    
        White_Blood_cell_count = st.number_input("White Blood cell count (thousand per microliter)")
        st.markdown("""
                    <div>
                    <h4 style ="color:green;text-align:left;"></h4>
                    </div>"""
                , unsafe_allow_html=True)
        Blood_test_result = st.selectbox("Blood test result", ('normal', 'slightly abnormal', 'inconclusive', 'abnormal'))
        Symptom_1 = st.selectbox("Symptom 1", (0, 1))
        Symptom_2 = st.selectbox("Symptom 2", (0, 1))
        Symptom_3 = st.selectbox("Symptom 3", (0, 1))
        Symptom_4 = st.selectbox("Symptom 4", (0, 1))
        Symptom_5 = st.selectbox("Symptom 5", (0, 1))
        sum_of_Mom_dad_age_avg = (Mother_age + Father_age)/2
        #st.markdown(sum_of_Mom_dad_age_avg)
        sum_of_Mom_dad_age_avg = st.number_input("sum_of_Mom_dad_age_avg")
        total_symptom = (Symptom_1 + Symptom_2 + Symptom_3 + Symptom_4 + Symptom_5)/5
        #st.markdown(total_symptom)
        total_symptom = st.number_input("total_symptom")

        # When Predict is clicked:
        if st.button("Predict"):
            result1 = prediction1(Patient_Age, Genes_in_mother_side, Inherited_from_father, Maternal_gene, 
                                 Paternal_gene, Blood_cell_count, Mother_age, Father_age, Status, 
                                 Respiratory_Rate, Heart_Rate, Test_1, Test_2, Test_3, Test_4, Test_5, 
                                 Parental_consent, Follow_up, Gender, Birth_asphyxia, Autopsy_shows_birth_defect, 
                                 Folic_acid_details, H_O_serious_maternal_illness, H_O_radiation_exposure, 
                                 H_O_substance_abuse, Assisted_conception, History_of_anomalies_in_prev_pregnancies, 
                                 No_of_previous_abortion, Birth_defects, White_Blood_cell_count, Blood_test_result, 
                                 Symptom_1, Symptom_2, Symptom_3, Symptom_4, Symptom_5
                                 , sum_of_Mom_dad_age_avg, total_symptom)
            st.success(f"Your Prediction : {result1}")
        
    
    elif choose == "Performance":
        col1, col2 = st.columns( [0.8, 0.2])
        with col1:               # To display the header text using css style
            st.markdown(""" <style> .font {
                font-size:35px ; font-family: 'Cooper Black'; color: Black;} 
                </style> """, unsafe_allow_html=True)
            st.markdown('<p class="font">Performance</p>', unsafe_allow_html=True)    
            # To display brand log
    
        st.write("Algorithm:  Support Vector Machine & XGBoost)")

        st.write("Accuracy Score 92.65")             
        plogo = Image.open(r'static/images/train2.png')
        plogo2 = Image.open(r'static/images/train1.png')
        st.image(plogo, width=700 )
        st.image(plogo2, width=700 )

if __name__ == '__main__':
    main()