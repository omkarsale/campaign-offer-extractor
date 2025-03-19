# Campaign Offer Extractor

A Streamlit application that extracts and categorizes campaign offers from campaign names and ad titles.

## Features

- Upload Excel/CSV files containing campaign data
- Automatically categorize campaigns into:
  - Medical Conditions
  - Health Concerns
  - General Categories (Education, Finance, Technology, Automobile, Concerts/Tour)
- Add or remove custom categories
- View statistics and distribution of campaign offers
- Download processed results

## Setup

1. Clone this repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Open the application in your web browser
2. Upload your Excel/CSV file containing campaign data
3. The file should have columns named either "Campaign Name" or "Campaign"
4. View the results and statistics
5. Download the processed file with categorized campaign offers

## Customization

Use the "Manage Categories" tab to:
- Add new categories to any category type
- Remove existing categories
- View current categories 