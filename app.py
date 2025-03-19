import streamlit as st
import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    # Remove the ID part in square brackets
    text = re.sub(r'\s*\[\d+\]$', '', str(text))
    # Remove special characters and extra spaces
    text = re.sub(r'[|_-]', ' ', text)
    return text.strip()

def simplify_condition(condition):
    """Simplify medical conditions by removing generic prefixes and standardizing names"""
    if not condition:
        return condition
        
    # Remove generic prefixes and suffixes
    generic_patterns = [
        # Prefixes
        r'^(?:Chronic|Acute|Advanced|Early Stage|Metastatic|Late Stage|Early|Severe)\s+',
        r'^(?:Learn About|About|Information About|Resources for)\s+',
        # Suffixes
        r'\s+(?:Resources|Information|Site|Visit|Official|Website|Treatment Option|Treatment|Option|Patient|Patients|Symptoms).*$',
        r'\s+(?:Injection|Cream|Medication|Drug|Therapy).*$'
    ]
    
    for pattern in generic_patterns:
        condition = re.sub(pattern, '', condition, flags=re.IGNORECASE)
    
    # Standardize cancer types
    if re.search(r'cancer', condition, re.IGNORECASE):
        cancer_match = re.search(r'([A-Za-z\s]+?)\s*cancer', condition, re.IGNORECASE)
        if cancer_match:
            cancer_type = cancer_match.group(1).strip()
            if cancer_type:
                return f"{cancer_type} Cancer"
            return "Cancer"
            
    # Standardize disease names
    if re.search(r'disease', condition, re.IGNORECASE):
        disease_match = re.search(r'([A-Za-z\s]+?)\s*disease', condition, re.IGNORECASE)
        if disease_match:
            disease_type = disease_match.group(1).strip()
            if disease_type:
                return f"{disease_type} Disease"
    
    return condition.strip()

def is_tech_related(text):
    tech_patterns = [
        r'tech(?:nology)?',
        r'connected',
        r'smart\s*(?:device|home|system)',
        r'digital',
        r'software',
        r'hardware',
        r'app(?:lication)?s?',
        r'internet',
        r'wireless',
        r'electronic',
        r'automation'
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in tech_patterns)

def is_finance_related(text):
    finance_patterns = [
        r'financ(?:e|ial)',
        r'bank(?:ing)?',
        r'invest(?:ment)?s?',
        r'money',
        r'credit',
        r'loan',
        r'mortgage',
        r'insurance',
        r'savings',
        r'debt',
        r'tax(?:es)?'
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in finance_patterns)

def is_auto_related(text):
    auto_patterns = [
        r'car[s]?',
        r'auto(?:mobile)?s?',
        r'vehicle[s]?',
        r'dealer(?:ship)?',
        r'driving',
        r'suv',
        r'truck'
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in auto_patterns)

def is_education_related(text):
    education_patterns = [
        r'edu(?:cation)?',
        r'school',
        r'college',
        r'university',
        r'degree',
        r'course',
        r'training',
        r'learn(?:ing)?',
        r'study',
        r'student'
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in education_patterns)

def is_entertainment_related(text):
    entertainment_patterns = [
        r'concert[s]?',
        r'tour[s]?',
        r'show[s]?',
        r'music',
        r'performance[s]?',
        r'festival[s]?',
        r'stubhub',
        r'ticket[s]?'
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in entertainment_patterns)

def extract_medical_condition(text):
    """Extract medical conditions that could be campaign offers"""
    # Common patterns for medical conditions
    condition_patterns = [
        # Disease patterns
        r'(?:about|learn about|have|living with|diagnosed with)\s+([A-Z][a-zA-Z\s\']+(?:Disease|Disorder|Cancer|osis|itis|emia|oma))',
        r'([A-Z][a-zA-Z\s\']+(?:Disease|Disorder|Cancer|osis|itis|emia|oma))\s*(?:-|:|$)',
        # Specific conditions
        r'(?:^|\s)((?:Alzheimer\'s|COPD|Asthma|Eczema|Psoriasis|Myelofibrosis|Multiple Sclerosis|MS|Graft Versus Host|Lupus|Hypersomnia|EDS)(?:\s+(?:Disease|Condition|Patient))?)',
        # Prefixed conditions
        r'(?:^|\s)((?:Metastatic|Advanced|Early Stage|Chronic|Severe|Idiopathic)\s+[A-Z][a-zA-Z\s\']+)'
    ]
    
    for pattern in condition_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            condition = match.group(1).strip()
            return simplify_condition(condition)
    
    # Check for specific medical terms without disease/disorder suffix
    specific_conditions = [
        r'(?:^|\s)(Lupus)\b',
        r'(?:^|\s)((?:Idiopathic\s+)?Hypersomnia)\b',
        r'(?:^|\s)(EDS)\b'
    ]
    
    for pattern in specific_conditions:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
            
    return None

def extract_campaign_offer(row):
    campaign_name = clean_text(row.get('Campaign Name', ''))
    ad_title = clean_text(row.get('Ad Title', ''))
    
    # Combine both texts for checking, prioritizing campaign name
    texts_to_check = [t for t in [campaign_name, ad_title] if t]
    
    # Check for entertainment/concerts/tours first
    for text in texts_to_check:
        if is_entertainment_related(text):
            return 'Concerts/Tour'
    
    # Check for medical conditions
    for text in texts_to_check:
        medical_condition = extract_medical_condition(text)
        if medical_condition:
            return medical_condition
    
    # Check for other categories
    for text in texts_to_check:
        if is_auto_related(text):
            return 'Automobile'
        if is_education_related(text):
            return 'Education'
        if is_tech_related(text):
            return 'Technology'
        if is_finance_related(text):
            return 'Finance'
    
    # Dictionary of specific conditions and their indicators
    offer_patterns = {
        'Blood Sugar': [
            r'blood\s*sugar',
            r'glucose',
            r'gluco',
            r'diabetes',
            r'a1c',
            r'insulin'
        ],
        'Weight Loss': [
            r'weight\s*loss',
            r'diet',
            r'slim',
            r'fat\s*burn',
            r'metabolism',
            r'keto',
            r'obesity'
        ],
        'Joint Pain': [
            r'joint\s*pain',
            r'arthritis',
            r'knee\s*pain',
            r'back\s*pain',
            r'inflammation'
        ],
        'Blood Pressure': [
            r'blood\s*pressure',
            r'hypertension',
            r'systolic',
            r'diastolic'
        ],
        'Sleep': [
            r'sleep\s*(?:aid|help|support)',
            r'insomnia',
            r'melatonin'
        ],
        'Hair Growth': [
            r'hair\s*(?:growth|loss)',
            r'baldness',
            r'alopecia',
            r'regrowth'
        ],
        'Memory': [
            r'memory',
            r'cognitive',
            r'brain\s*health',
            r'focus',
            r'concentration'
        ],
        'Prostate': [
            r'prostate',
            r'urinary',
            r'bladder'
        ],
        'Vision': [
            r'vision',
            r'eye\s*health',
            r'macular',
            r'retina'
        ],
        'Hearing': [
            r'hearing',
            r'tinnitus',
            r'ear\s*health'
        ],
        'Schizophrenia': [
            r'schizophrenia',
            r'schizoaffective',
            r'psychosis'
        ],
        'Lung Cancer': [
            r'lung\s*cancer',
            r'metastatic\s*lung',
            r'nsclc',
            r'sclc'
        ],
        'COPD': [
            r'copd',
            r'chronic\s*obstructive\s*pulmonary\s*disease',
            r'emphysema'
        ],
        'Asthma': [
            r'asthma',
            r'breathing\s*difficulty',
            r'bronchial'
        ],
        'Eczema': [
            r'eczema',
            r'atopic\s*dermatitis'
        ],
        "Alzheimer's": [
            r'alzheimer',
            r'dementia'
        ],
        'Myelofibrosis': [
            r'myelofibrosis',
            r'bone\s*marrow'
        ],
        'Graft Versus Host Disease': [
            r'graft\s*versus\s*host',
            r'gvhd'
        ]
    }
    
    # Use existing offer patterns dictionary
    for text in texts_to_check:
        text = text.lower()
        for offer, patterns in offer_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return offer
    
    return 'NA'

def process_dataframe(df, campaign_col):
    """Process the dataframe and ensure same campaign gets same offer"""
    # First pass: extract offers
    df['Campaign Offer'] = df.apply(extract_campaign_offer, axis=1)
    
    # Create a mapping of campaign names to their non-NA offers
    campaign_offers = {}
    for _, row in df.iterrows():
        campaign = row[campaign_col]
        offer = row['Campaign Offer']
        if offer != 'NA' and campaign not in campaign_offers:
            campaign_offers[campaign] = offer
    
    # Second pass: apply consistent offers across campaigns
    for campaign, offer in campaign_offers.items():
        df.loc[df[campaign_col] == campaign, 'Campaign Offer'] = offer
    
    return df

def main():
    st.title('Campaign Offer Extractor')
    st.write('Upload your Excel/CSV file containing campaign names and ad titles to extract campaign offers.')
    
    uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'csv'])
    
    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Check for required columns
            campaign_col = None
            ad_title_col = None
            
            # Check for Campaign Name column
            campaign_name_variants = ['Campaign Name', 'Campaign', 'campaign name', 'campaign']
            ad_title_variants = ['Ad Title', 'Ad_Title', 'ad title', 'ad_title', 'Title']
            
            for col in campaign_name_variants:
                if col in df.columns:
                    campaign_col = col
                    break
                    
            for col in ad_title_variants:
                if col in df.columns:
                    ad_title_col = col
                    break
            
            if campaign_col is None:
                st.error("Please ensure your file has a column named 'Campaign Name' or 'Campaign'")
                return
            
            # Process the dataframe
            df = process_dataframe(df, campaign_col)
            
            # Display results
            st.write("### Results")
            st.dataframe(df)
            
            # Show statistics
            offer_stats = df['Campaign Offer'].value_counts()
            st.write("### Campaign Offer Distribution")
            st.bar_chart(offer_stats)
            
            # Download button for processed file
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Processed File",
                csv,
                "processed_campaigns.csv",
                "text/csv",
                key='download-csv'
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 