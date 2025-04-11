from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Define the model path - use relative path for better portability
model_path = 'career_recommendation_model.pkl'

# Load model with error handling
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Column names for the features
columns = [f'Q{i}' for i in range(1, 21)]  # Q1 to Q20

# Career explanations
career_descriptions = {
    "Software Developer": "Software developers design, build, and maintain computer programs. Your assessment shows strong analytical thinking and problem-solving abilities, which are essential in this field.",
    
    "Data Scientist": "Data scientists analyze and interpret complex data to help organizations make better decisions. Your results indicate strong analytical skills and comfort with data analysis.",
    
    "Marketing Manager": "Marketing managers develop strategies to promote products and services. Your profile shows creativity combined with business acumen, which suits this role well.",
    
    "Healthcare Professional": "Healthcare professionals provide medical care and support to patients. Your assessment indicates strong empathy and helping orientation, which are fundamental in this field.",
    
    "Teacher/Educator": "Teachers inspire and educate students of various ages. Your results suggest strong communication skills and a desire to help others grow and develop.",
    
    "Financial Analyst": "Financial analysts assess financial data and make recommendations for investments and business decisions. Your assessment shows strong numerical reasoning and analytical thinking.",
    
    "Project Manager": "Project managers coordinate resources and people to complete projects. Your profile indicates good organizational and leadership skills necessary for this role.",
    
    "Human Resources Manager": "HR managers oversee employee relations, recruitment, and organizational development. Your assessment shows strong interpersonal skills and interest in human behavior.",
    
    "Entrepreneur": "Entrepreneurs start and manage their own businesses. Your results suggest creativity, risk tolerance, and independent thinking that entrepreneurs typically possess.",
    
    "Graphic Designer": "Graphic designers create visual concepts for various media. Your assessment indicates strong creative thinking and artistic abilities."
}

# Default explanation for careers not in our dictionary
default_explanation = "This career aligns with your personality traits and preferences based on the psychometric assessment patterns of professionals in this field."

@app.route('/')
def home():
    questions = [
        "I enjoy solving complex logical and mathematical problems",
        "I find satisfaction in helping others overcome difficulties",
        "I like creating things with my hands or physical tools",
        "I prefer creative and artistic activities",
        "I'm interested in how businesses operate and generate profit",
        "I enjoy understanding how technology and complex systems work",
        "I like studying natural systems and living organisms",
        "I'm comfortable working with and analyzing large amounts of data",
        "I like environments that are structured and organized",
        "I prefer working independently with minimal oversight",
        "I thrive in collaborative, team-oriented settings",
        "I enjoy leading projects and coordinating others' work",
        "I like roles that require precise attention to detail",
        "I enjoy fast-paced environments with changing priorities",
        "I'm interested in understanding human behavior and psychology",
        "I enjoy communicating ideas through presentations or teaching",
        "I prefer work that contributes directly to society's well-being",
        "I'm motivated by challenges and achieving difficult goals",
        "I value work-life balance over career advancement",
        "I enjoy taking calculated risks and making important decisions"
    ]
    return render_template('index.html', questions=questions)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return render_template('result.html', error="Model could not be loaded. Please contact the administrator.")
        
        career_label_to_name = {
    "Career_0": "Data Scientist",
    "Career_1": "Psychologist",
    "Career_2": "Software Engineer",
    "Career_3": "Artist",
    "Career_4": "Doctor",
    "Career_5": "Teacher",
    "Career_6": "Business Analyst",
    "Career_7": "Engineer",
    "Career_8": "Biologist",
    "Career_9": "Entrepreneur",
    "Career_10": "Journalist",
    "Career_11": "Marketing Manager",
    "Career_12": "Nurse",
    "Career_13": "Accountant",
    "Career_14": "UX Designer",
    "Career_15": "Financial Analyst",
    "Career_16": "Architect",
    "Career_17": "Social Worker",
    "Career_18": "IT Support Specialist",
    "Career_19": "Chef",
    "Career_20": "Human Resources Manager",
    "Career_21": "Research Scientist",
    "Career_22": "Lawyer",
    "Career_23": "Electrician",
    "Career_24": "Graphic Designer",
    "Career_25": "Product Manager",
    "Career_26": "Civil Engineer",
    "Career_27": "Veterinarian",
    "Career_28": "Consultant",
    "Career_29": "Pharmacist",
    "Career_30": "Event Planner",
    "Career_31": "Mechanical Engineer",
    "Career_32": "Environmental Scientist",
    "Career_33": "Real Estate Agent",
    "Career_34": "Physical Therapist",
    "Career_35": "Supply Chain Manager",
    "Career_36": "Web Developer",
    "Career_37": "Customer Service Rep",
    "Career_38": "Project Manager",
    "Career_39": "College Professor",
    "Career_40": "Dental Hygienist",
    "Career_41": "Pilot",
    "Career_42": "Cybersecurity Analyst",
    "Career_43": "Fashion Designer",
    "Career_44": "Speech Therapist",
    "Career_45": "Investment Banker",
    "Career_46": "Photographer",
    "Career_47": "Clinical Psychologist",
    "Career_48": "Game Developer",
    "Career_49": "Urban Planner",
    "Career_50": "Flight Attendant",
    "Career_51": "Robotics Engineer",
    "Career_52": "Environmental Lawyer",
    "Career_53": "Interior Designer",
    "Career_54": "Music Teacher",
    "Career_55": "Data Analyst",
    "Career_56": "Dentist",
    "Career_57": "Fitness Trainer",
    "Career_58": "Aerospace Engineer",
    "Career_59": "Content Creator",
    "Career_60": "Occupational Therapist",
    "Career_61": "Financial Planner",
    "Career_62": "App Developer",
    "Career_63": "Marriage Counselor",
    "Career_64": "Geologist",
    "Career_65": "Chef de Cuisine",
    "Career_66": "Public Relations Specialist",
    "Career_67": "Neurologist",
    "Career_68": "Architect (Software)",
    "Career_69": "Social Media Manager",
    "Career_70": "Physicist",
    "Career_71": "Landscape Designer",
    "Career_72": "Emergency Medical Technician",
    "Career_73": "Air Traffic Controller",
    "Career_74": "Historian",
    "Career_75": "Hotel Manager",
    "Career_76": "Nuclear Engineer",
    "Career_77": "Marine Biologist",
    "Career_78": "Art Director",
    "Career_79": "Dental Assistant",
    "Career_80": "Mechanical Technician",
    "Career_81": "Special Education Teacher",
    "Career_82": "Technical Writer",
    "Career_83": "Pharmaceutical Sales",
    "Career_84": "Forensic Scientist",
    "Career_85": "Athletic Trainer",
    "Career_86": "Database Administrator",
    "Career_87": "Interior Decorator",
    "Career_88": "Nurse Practitioner",
    "Career_89": "Financial Controller",
    "Career_90": "UI Developer",
    "Career_91": "School Counselor",
    "Career_92": "Geophysicist",
    "Career_93": "Executive Chef",
    "Career_94": "Digital Marketing Specialist",
    "Career_95": "Cardiologist",
    "Career_96": "DevOps Engineer",
    "Career_97": "Public Speaker",
    "Career_98": "Quantum Physicist",
    "Career_99": "Floral Designer",
    "Career_100": "Speech-Language Pathologist",
    "Career_101": "Investment Analyst",
    "Career_102": "3D Artist",
    "Career_103": "Clinical Nurse Specialist",
    "Career_104": "Tax Accountant",
    "Career_105": "UX Researcher",
    "Career_106": "Marriage Therapist",
    "Career_107": "Astronomer",
    "Career_108": "Restaurant Manager",
    "Career_109": "SEO Specialist",
    "Career_110": "Surgical Technologist",
    "Career_111": "Machine Learning Engineer",
    "Career_112": "Event Host",
    "Career_113": "Meteorologist",
    "Career_114": "Fashion Merchandiser",
    "Career_115": "Physical Education Teacher",
    "Career_116": "Systems Administrator",
    "Career_117": "Interior Architect",
    "Career_118": "Genetic Counselor",
    "Career_119": "Actuary",
    "Career_120": "Video Game Artist",
    "Career_121": "Speech Coach",
    "Career_122": "Astronaut"
}

        
        # Get user responses
        responses = [int(request.form[f'Q{i+1}']) for i in range(20)]
        df = pd.DataFrame([responses], columns=columns)
        
        # Get prediction probabilities
        proba = model.predict_proba(df)[0]
        classes = model.classes_

        # Get top 5 recommendations
        top_n = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)[:5]
        
        # Format recommendations with explanations
        recommendations = []
        for job, score in top_n:
            # Get explanation for the career (use default if not found)
            explanation = career_descriptions.get(job, default_explanation)
            
            # Add strength-based insights based on user responses
            high_scores = []
            for i, response in enumerate(responses):
                if response >= 4:  # User scored high (4 or 5)
                    high_scores.append(i)
            
            # Map question indices to skills
            skills_map = {
                0: "analytical thinking",
                1: "helping others",
                2: "working with hands",
                3: "creativity",
                4: "business acumen",
                5: "technical understanding",
                6: "interest in natural systems",
                7: "data analysis",
                8: "organizational skills",
                9: "independent work",
                10: "teamwork",
                11: "leadership",
                12: "attention to detail",
                13: "adaptability",
                14: "understanding human behavior",
                15: "communication",
                16: "social contribution",
                17: "challenge-seeking",
                18: "work-life balance",
                19: "decision-making"
            }
            
            # Pick up to 3 relevant high-scoring skills
            relevant_skills = [skills_map[i] for i in high_scores[:3]]
            if relevant_skills:
                explanation += f"<br><br>Your strengths in {', '.join(relevant_skills)} are particularly valuable in this career path."
            
            recommendations.append({
                'career': career_label_to_name.get(job, job),
                'confidence': round(score * 1000, 2),
                'explanation': explanation
            })

        return render_template('result.html', recommendations=recommendations)

    except Exception as e:
        error_message = f"Error processing prediction: {str(e)}"
        print(error_message)
        return render_template('result.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)