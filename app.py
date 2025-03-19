import streamlit as st
import pandas as pd
import re

# Initialize session state for custom categories
if 'custom_categories' not in st.session_state:
    st.session_state.custom_categories = {
        'Medical Conditions': [
            'Myelofibrosis', 'Breast Cancer', 'Sclerosis', 'Schizophrenia', 'COPD',
            'Lung Cancer', 'Lupus', 'Alzheimer', 'Dementia', 'Blood Cancer', 'Joint Pain',
            'Prostate', 'Asthma', 'Eczema', 'Myeloma', 'Kidney Disease',
            'Hypersomnia', 'IH Diagnosis', 'EDS', 'Back Pain', 'Breast Cancer'
        ],
        'Health Concerns': [
            'Weight Loss', 'Blood Pressure', 'Joint Pain', 'Blood Sugar', 'Memory', 'Hearing',
            'Prostate', 'Aligner System'
        ],
        'General Categories': [
            'Education', 'Finance', 'Technology', 'Automobile', 'Concerts/Tour'
        ]
    }

def clean_text(text):
    if pd.isna(text):
        return ""
    # Remove the ID part in square brackets
    text = re.sub(r'\s*\[\d+\]$', '', str(text))
    # Remove special characters and extra spaces
    text = re.sub(r'[|_-]', ' ', text)
    return text.strip()

def decode_medical_abbreviations(text):
    """Decode common medical abbreviations"""
    abbreviations = {
        r'\bMBC\b': 'Metastatic Breast Cancer',
        r'\bCLL\b': 'Chronic Lymphocytic Leukemia',
        r'\bMS\b': 'Multiple Sclerosis',
        r'\bRA\b': 'Rheumatoid Arthritis',
        r'\bIBD\b': 'Inflammatory Bowel Disease',
        r'\bCOPD\b': 'Chronic Obstructive Pulmonary Disease',
        r'\bAML\b': 'Acute Myeloid Leukemia',
        r'\bCML\b': 'Chronic Myeloid Leukemia',
        r'\bITP\b': 'Immune Thrombocytopenia',
        r'\bPNH\b': 'Paroxysmal Nocturnal Hemoglobinuria',
        r'\bSLE\b': 'Systemic Lupus Erythematosus',
        r'\bT2D\b': 'Type 2 Diabetes',
        r'\bHER2\+?\b': 'HER2-Positive Breast Cancer',
        r'\bNSCLC\b': 'Non-Small Cell Lung Cancer',
        r'\bSCLC\b': 'Small Cell Lung Cancer',
        r'\bPsA\b': 'Psoriatic Arthritis',
        r'\bAS\b': 'Ankylosing Spondylitis',
        r'\bUC\b': 'Ulcerative Colitis',
        r'\bCD\b': 'Crohn\'s Disease',
        r'\bPV\b': 'Polycythemia Vera',
        r'\bET\b': 'Essential Thrombocythemia',
        r'\bMF\b': 'Myelofibrosis',
        r'\bMDS\b': 'Myelodysplastic Syndrome',
        r'\bAD\b': 'Atopic Dermatitis',
        r'\bGVHD\b': 'Graft Versus Host Disease',
        r'\bEDS\b': 'Ehlers-Danlos Syndrome'
    }
    
    decoded_text = text
    for abbr, full in abbreviations.items():
        decoded_text = re.sub(abbr, full, decoded_text, flags=re.IGNORECASE)
    return decoded_text

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
        r'\s+(?:Injection|Cream|Medication|Drug|Therapy).*$',
        r'\s+(?:Learn More|Find Out More|Get Info).*$'
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
    ]
}

def validate_condition(text, extracted_condition):
    """Simple rule-based validation similar to LLM reasoning"""
    
    # List of words that indicate it's likely about a medical condition
    medical_indicators = [
        'treatment', 'therapy', 'medication', 'symptoms', 'disease', 'disorder',
        'syndrome', 'patient', 'diagnosed', 'living with', 'managing', 'caregiver',
        'clinical', 'medicine', 'health', 'condition'
    ]
    
    # List of common medical conditions to validate against
    known_conditions = {
        'ADHD': ['adhd', 'attention deficit'],
        'Myelofibrosis': ['myelofibrosis', 'myelo'],
        'Hemophilia': ['hemophilia'],
        'Multiple Sclerosis': ['multiple sclerosis', 'ms'],
        'Lupus': ['lupus', 'sle'],
        'Schizophrenia': ['schizophrenia'],
        'Asthma': ['asthma'],
        'COPD': ['copd', 'chronic obstructive'],
        'Psoriasis': ['psoriasis'],
        'Alzheimer': ['alzheimer'],
        'Cancer': ['cancer'],
        'Diabetes': ['diabetes'],
        'Arthritis': ['arthritis'],
        'Eczema': ['eczema'],
        'Depression': ['depression'],
        'Anxiety': ['anxiety'],
        'Migraine': ['migraine'],
        'Epilepsy': ['epilepsy'],
        'Parkinson': ['parkinson'],
        'Fibromyalgia': ['fibromyalgia']
    }
    
    # Convert text to lowercase for comparison
    text_lower = text.lower()
    
    # Check if the text contains medical indicators
    has_medical_context = any(indicator in text_lower for indicator in medical_indicators)
    
    # If we have a condition, validate it against known conditions
    if extracted_condition:
        # Check if the extracted condition is in our known conditions list
        for condition, variants in known_conditions.items():
            if any(variant in extracted_condition.lower() for variant in variants):
                return condition
            
        # If it's not in known conditions but has medical context, it might be valid
        if has_medical_context:
            return extracted_condition
    
    return None

def extract_clean_condition(text):
    """Extract and clean medical condition names"""
    # First, try to find standalone condition names
    condition_patterns = [
        # Match condition names that appear after certain prefixes
        r'(?:about|for|with)\s+([A-Z][a-zA-Z\s\']+?)(?:\s+(?:Resources|Information|Treatment|Therapy|Patient|Symptoms|Disease|Disorder)s?)?(?:\s|$)',
        # Match condition names that appear before certain suffixes
        r'([A-Z][a-zA-Z\s\']+?)(?:\s+(?:Resources|Information|Treatment|Therapy|Patient|Symptoms|Disease|Disorder)s?)?(?:\s|$)',
        # Match specific conditions
        r'\b(ADHD|MS|COPD|IBD|RA|SLE|CLL|AML|CML|MDS|PNH|ITP)\b',
        # Match conditions with Disease/Syndrome/Disorder
        r'([A-Z][a-zA-Z\s\']+?)\s+(?:Disease|Syndrome|Disorder)\b'
    ]
    
    for pattern in condition_patterns:
        match = re.search(pattern, text)
        if match:
            condition = match.group(1).strip()
            # Remove any remaining suffixes or prefixes
            condition = re.sub(r'\s+(?:Resources|Information|Treatment|Therapy|Patient|Symptoms).*$', '', condition)
            condition = re.sub(r'^(?:About|For|With)\s+', '', condition)
            return condition
    
    return None

def add_custom_category():
    category_type = st.selectbox(
        "Select Category Type",
        ["Medical Conditions", "Health Concerns", "General Categories"]
    )
    
    new_category = st.text_input("Enter new category name:")
    if st.button("Add Category") and new_category:
        if new_category not in st.session_state.custom_categories[category_type]:
            st.session_state.custom_categories[category_type].append(new_category)
            st.success(f"Added {new_category} to {category_type}")
        else:
            st.warning("Category already exists!")

def remove_custom_category():
    category_type = st.selectbox(
        "Select Category Type to Remove From",
        ["Medical Conditions", "Health Concerns", "General Categories"],
        key="remove_category_type"
    )
    
    if st.session_state.custom_categories[category_type]:
        category_to_remove = st.selectbox(
            "Select category to remove",
            st.session_state.custom_categories[category_type]
        )
        if st.button("Remove Category"):
            st.session_state.custom_categories[category_type].remove(category_to_remove)
            st.success(f"Removed {category_to_remove}")

def validate_general_category(text, category):
    """Smart validation for general categories based on context and domain analysis"""
    text_lower = text.lower()
    
    # Comprehensive domain indicators with weights
    category_indicators = {
        'Education': {
            'high_confidence': [
                r'degree\s*program',
                r'online\s*course',
                r'certification',
                r'university',
                r'college',
                r'diploma',
                r'scholarship',
                r'academic'
            ],
            'medium_confidence': [
                r'study',
                r'learn',
                r'training',
                r'education',
                r'school',
                r'class',
                r'course',
                r'career',
                r'skills'
            ],
            'context_words': [
                r'enroll',
                r'admission',
                r'student',
                r'program',
                r'curriculum',
                r'faculty',
                r'campus'
            ]
        },
        'Finance': {
            'high_confidence': [
                r'investment\s*(?:strategy|portfolio|plan)',
                r'stock\s*(?:market|trading|analysis)',
                r'financial\s*(?:planning|advisor|services)',
                r'mortgage\s*(?:rates|loan|refinance)',
                r'retirement\s*(?:planning|savings|fund)',
                r'credit\s*(?:card|score|report)',
                r'tax\s*(?:services|preparation|planning)'
            ],
            'medium_confidence': [
                r'invest',
                r'banking',
                r'stocks',
                r'mutual\s*funds',
                r'insurance',
                r'loan',
                r'mortgage',
                r'savings',
                r'financial'
            ],
            'context_words': [
                r'money',
                r'budget',
                r'wealth',
                r'income',
                r'market',
                r'trading',
                r'portfolio',
                r'interest\s*rate'
            ]
        },
        'Technology': {
            'high_confidence': [
                r'software\s*(?:development|solution)',
                r'cloud\s*(?:computing|storage|service)',
                r'artificial\s*intelligence',
                r'machine\s*learning',
                r'cyber\s*security',
                r'data\s*(?:analytics|science)',
                r'mobile\s*(?:app|application)',
                r'smart\s*(?:device|home|technology)'
            ],
            'medium_confidence': [
                r'tech',
                r'software',
                r'digital',
                r'automation',
                r'innovation',
                r'platform',
                r'solution',
                r'application',
                r'system'
            ],
            'context_words': [
                r'device',
                r'network',
                r'internet',
                r'wireless',
                r'computing',
                r'online',
                r'electronic',
                r'virtual'
            ]
        },
        'Automobile': {
            'high_confidence': [
                r'car\s*(?:dealer|dealership|sale)',
                r'auto\s*(?:repair|service|maintenance)',
                r'vehicle\s*(?:inspection|maintenance)',
                r'new\s*(?:car|vehicle)',
                r'certified\s*pre\s*owned',
                r'test\s*drive',
                r'car\s*(?:loan|insurance|financing)'
            ],
            'medium_confidence': [
                r'automotive',
                r'dealership',
                r'vehicle',
                r'car',
                r'auto',
                r'suv',
                r'truck',
                r'lease'
            ],
            'context_words': [
                r'driver',
                r'driving',
                r'engine',
                r'model',
                r'maintenance',
                r'repair',
                r'parts',
                r'service'
            ]
        }
    }
    
    if category in category_indicators:
        # Calculate confidence score
        score = 0
        patterns = category_indicators[category]
        
        # Check high confidence patterns (worth 3 points each)
        high_matches = sum(1 for pattern in patterns['high_confidence'] 
                         if re.search(pattern, text_lower))
        score += high_matches * 3
        
        # Check medium confidence patterns (worth 2 points each)
        medium_matches = sum(1 for pattern in patterns['medium_confidence'] 
                           if re.search(pattern, text_lower))
        score += medium_matches * 2
        
        # Check context words (worth 1 point each)
        context_matches = sum(1 for pattern in patterns['context_words'] 
                            if re.search(pattern, text_lower))
        score += context_matches
        
        # Return true if we have either:
        # - At least one high confidence match
        # - At least two medium confidence matches
        # - One medium confidence match and two context matches
        # - Total score of 4 or higher
        return (high_matches >= 1 or 
                medium_matches >= 2 or 
                (medium_matches >= 1 and context_matches >= 2) or 
                score >= 4)
    
    return False

def match_category(text):
    """Match text against custom categories in priority order with smart validation"""
    text_lower = text.lower()
    
    # First check Medical Conditions (highest priority)
    for condition in st.session_state.custom_categories['Medical Conditions']:
        if condition.lower() in text_lower:
            return condition
            
    # Then check Health Concerns
    for concern in st.session_state.custom_categories['Health Concerns']:
        if concern.lower() in text_lower:
            return concern
    
    # Finally check General Categories with smart validation
    # We'll check all categories and pick the one with the strongest match
    best_category = None
    max_score = 0
    
    for category in st.session_state.custom_categories['General Categories']:
        if category in ['Education', 'Finance', 'Technology', 'Automobile']:
            # Calculate match score for this category
            score = calculate_category_score(text, category)
            if score > max_score:
                max_score = score
                best_category = category
    
    if best_category and max_score >= 4:  # Minimum threshold for confidence
        return best_category
                
    return 'NA'

def calculate_category_score(text, category):
    """Calculate a confidence score for a general category match"""
    text_lower = text.lower()
    
    # Reuse the patterns from validate_general_category
    category_indicators = {
        'Education': {
            'high_confidence': [
                r'degree\s*program',
                r'online\s*course',
                r'certification',
                r'university',
                r'college',
                r'diploma',
                r'scholarship',
                r'academic'
            ],
            'medium_confidence': [
                r'study',
                r'learn',
                r'training',
                r'education',
                r'school',
                r'class',
                r'course',
                r'career',
                r'skills'
            ],
            'context_words': [
                r'enroll',
                r'admission',
                r'student',
                r'program',
                r'curriculum',
                r'faculty',
                r'campus'
            ]
        },
        'Finance': {
            'high_confidence': [
                r'investment\s*(?:strategy|portfolio|plan)',
                r'stock\s*(?:market|trading|analysis)',
                r'financial\s*(?:planning|advisor|services)',
                r'mortgage\s*(?:rates|loan|refinance)',
                r'retirement\s*(?:planning|savings|fund)',
                r'credit\s*(?:card|score|report)',
                r'tax\s*(?:services|preparation|planning)'
            ],
            'medium_confidence': [
                r'invest',
                r'banking',
                r'stocks',
                r'mutual\s*funds',
                r'insurance',
                r'loan',
                r'mortgage',
                r'savings',
                r'financial'
            ],
            'context_words': [
                r'money',
                r'budget',
                r'wealth',
                r'income',
                r'market',
                r'trading',
                r'portfolio',
                r'interest\s*rate'
            ]
        },
        'Technology': {
            'high_confidence': [
                r'software\s*(?:development|solution)',
                r'cloud\s*(?:computing|storage|service)',
                r'artificial\s*intelligence',
                r'machine\s*learning',
                r'cyber\s*security',
                r'data\s*(?:analytics|science)',
                r'mobile\s*(?:app|application)',
                r'smart\s*(?:device|home|technology)'
            ],
            'medium_confidence': [
                r'tech',
                r'software',
                r'digital',
                r'automation',
                r'innovation',
                r'platform',
                r'solution',
                r'application',
                r'system'
            ],
            'context_words': [
                r'device',
                r'network',
                r'internet',
                r'wireless',
                r'computing',
                r'online',
                r'electronic',
                r'virtual'
            ]
        },
        'Automobile': {
            'high_confidence': [
                r'car\s*(?:dealer|dealership|sale)',
                r'auto\s*(?:repair|service|maintenance)',
                r'vehicle\s*(?:inspection|maintenance)',
                r'new\s*(?:car|vehicle)',
                r'certified\s*pre\s*owned',
                r'test\s*drive',
                r'car\s*(?:loan|insurance|financing)'
            ],
            'medium_confidence': [
                r'automotive',
                r'dealership',
                r'vehicle',
                r'car',
                r'auto',
                r'suv',
                r'truck',
                r'lease'
            ],
            'context_words': [
                r'driver',
                r'driving',
                r'engine',
                r'model',
                r'maintenance',
                r'repair',
                r'parts',
                r'service'
            ]
        }
    }
    
    if category in category_indicators:
        score = 0
        patterns = category_indicators[category]
        
        # High confidence matches (3 points each)
        score += sum(3 for pattern in patterns['high_confidence'] 
                    if re.search(pattern, text_lower))
        
        # Medium confidence matches (2 points each)
        score += sum(2 for pattern in patterns['medium_confidence'] 
                    if re.search(pattern, text_lower))
        
        # Context matches (1 point each)
        score += sum(1 for pattern in patterns['context_words'] 
                    if re.search(pattern, text_lower))
        
        return score
    
    return 0

def extract_campaign_offer(row):
    campaign_name = clean_text(row.get('Campaign Name', ''))
    ad_title = clean_text(row.get('Ad Title', ''))
    
    # Combine texts for checking
    combined_text = ' '.join([t for t in [campaign_name, ad_title] if t])
    
    # Match against custom categories
    return match_category(combined_text)

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
    
    # Add tabs for file upload and category management
    tab1, tab2 = st.tabs(["Process Files", "Manage Categories"])
    
    with tab1:
        st.write('Upload your Excel/CSV file containing campaign names and ad titles.')
        uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'csv'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Check for required columns
                campaign_col = None
                for col in ['Campaign Name', 'Campaign', 'campaign name', 'campaign']:
                    if col in df.columns:
                        campaign_col = col
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
                
                # Download button
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
    
    with tab2:
        st.write("### Manage Categories")
        st.write("Add or remove categories to customize the matching process.")
        
        # Display current categories
        st.write("#### Current Categories:")
        for category_type, categories in st.session_state.custom_categories.items():
            st.write(f"**{category_type}:**")
            st.write(", ".join(sorted(categories)))
        
        st.write("---")
        
        # Add new category
        st.write("#### Add New Category")
        add_custom_category()
        
        st.write("---")
        
        # Remove category
        st.write("#### Remove Category")
        remove_custom_category()

if __name__ == "__main__":
    main() 