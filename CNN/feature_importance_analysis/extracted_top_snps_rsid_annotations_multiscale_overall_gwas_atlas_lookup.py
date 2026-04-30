#!/usr/bin/env python3
import argparse
import pandas as pd
import requests
import re
from tqdm import tqdm
import time

PHEWAS_PAGE = "https://atlas.ctglab.nl/PheWAS"
PHEWAS_URL = "https://atlas.ctglab.nl/PheWAS/getData"


def trait_matches(trait: str) -> bool:
    """Return True if trait is a direct cancer or diabetes diagnosis."""
    t = trait.lower()
    
    # Exclude anything that's not a direct diagnosis
    exclude_patterns = [
        "non-cancer",
        "non-diabetic"
        "age at",
        "father:",
        "mother:",
        "family",
        "siblings",
        "screening",
        "mammogram",
        "medication",
        "drugs",
        "ever had",
        "history of",
        "Epithelial",
        "skin",
        "self-reported"
    ]
    
    for exclude in exclude_patterns:
        if exclude in t:
            return False
    
    cancer_indicators = [
        " cancer",  # Space before to avoid "non-cancer"
        "carcinoma", 
        "tumor",
        "tumour",
        "neoplasm",
        "malignant",
        "leukemia",
        "leukaemia",
        "lymphoma",
        "melanoma",
        "sarcoma",
        "glioma",
        "myeloma"
    ]
    
    # For diabetes: look for actual diabetes diagnoses
    diabetes_indicators = [
        "diabetes mellitus",
        "diabetes (diagnosed",
        "type 1 diabetes",
        "type 2 diabetes",
        "diabetic",
        "t1d",
        "t2d",
        "diabetic retinopathy",
        "diabetic neuropathy",
        "icd10: e10",  # Type 1 diabetes ICD code
        "icd10: e11"   # Type 2 diabetes ICD code
    ]
    
    has_cancer = any(term in t for term in cancer_indicators)
    has_diabetes = any(term in t for term in diabetes_indicators)
    
    return has_cancer or has_diabetes

def get_session_and_token():
    """Get a session and CSRF token from the PheWAS page."""
    session = requests.Session()
    
    try:
        response = session.get(PHEWAS_PAGE, timeout=10)
        
        if response.status_code != 200:
            print(f"Failed to get session: {response.status_code}")
            return None, None
        
        # Extract CSRF token from meta tag
        match = re.search(r'<meta name="csrf-token" content="([^"]+)"', response.text)
        if match:
            csrf_token = match.group(1)
            return session, csrf_token
        
        print("Could not find CSRF token")
        return session, None
        
    except Exception as e:
        print(f"Error getting session: {e}")
        return None, None


def query_phewas(session, csrf_token, rsid, max_pvalue=0.05):
    """Query the PheWAS endpoint for a single rsID."""
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'X-Requested-With': 'XMLHttpRequest',
        'X-CSRF-TOKEN': csrf_token,
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Referer': PHEWAS_PAGE,
        'Origin': 'https://atlas.ctglab.nl',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
    }
    
    data = {
        'text': rsid,
        'id': '',
        'maxP': str(max_pvalue)
    }
    
    try:
        response = session.post(PHEWAS_URL, data=data, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"Error for {rsid}: Status {response.status_code}")
            return None
            
        return response.json()
        
    except Exception as e:
        print(f"Error for {rsid}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Lookup SNP associations from GWAS Atlas PheWAS.")

    parser.add_argument("-input", default="/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/m4/feat_imp/overall_top_50000_snps_integrated_gradients_overall_test_set_annotated.csv", 
                       help="Input CSV file containing SNP IDs.")
    parser.add_argument("-output", default="/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/m4/feat_imp/overall_top_50000_snps_integrated_gradients_overall_test_set_annotated_atlas_lookup.csv", 
                       help="Output CSV file for all associations.")
    parser.add_argument("-column", default="snp_id", help="Column name containing rsIDs (default: snp_id).")
    parser.add_argument("-maxp", default=0.05, type=float, help="Maximum p-value threshold (default: 0.05).")

    args = parser.parse_args()

    # Get session and CSRF token first
    print("Getting session and CSRF token...")
    session, csrf_token = get_session_and_token()
    
    if not session or not csrf_token:
        print("Failed to get session. Exiting.")
        return

    print("✓ Session obtained\n")

    # Load input CSV - keep ALL columns
    df_input = pd.read_csv(args.input)
    print(f"Loaded input file with {len(df_input)} rows and {len(df_input.columns)} columns")
    print(f"Columns: {list(df_input.columns)}\n")

    if args.column not in df_input.columns:
        raise ValueError(f"Column '{args.column}' not found in input file.")

    # Get unique rsIDs to query (avoid duplicate API calls)
    rsids = df_input[args.column].dropna().astype(str).str.strip().unique().tolist()

    all_rows = []
    filtered_rows = []

    print(f"Processing {len(rsids)} unique SNPs...\n")

    for rsid in tqdm(rsids):
        result = query_phewas(session, csrf_token, rsid, args.maxp)
        
        if not result or 'data' not in result:
            time.sleep(0.3)
            continue

        # Parse response - data is a list of lists
        associations = result['data']
        
        for assoc in associations:
            # Each association is a list: [atlas_id, p_value, pmid, year, domain, trait, n, ea, nea]
            if len(assoc) < 9:
                continue
                
            atlas_id = assoc[0]
            pval = assoc[1]
            pmid = assoc[2]
            year = assoc[3]
            category = assoc[4]
            trait = assoc[5]
            n = assoc[6]
            ea = assoc[7]
            nea = assoc[8]

            row = {
                "rsid": rsid,
                "atlas_id": atlas_id,
                "trait": trait,
                "category": category,
                "pval": pval,
                "pmid": pmid,
                "year": year,
                "N": n,
                "EA": ea,
                "NEA": nea
            }

            all_rows.append(row)

            if trait_matches(trait):
                filtered_rows.append(row)
        
        time.sleep(0.3)  # Rate limiting

    # Convert to DataFrames
    if all_rows:
        df_all_assoc = pd.DataFrame(all_rows)
        
        # Merge with original input data to keep all columns
        # This creates one row per association, with all original columns duplicated
        df_merged = df_all_assoc.merge(
            df_input, 
            left_on='rsid', 
            right_on=args.column, 
            how='left'
        )
        
        # Save merged data
        df_merged.to_csv(args.output, index=False)
        print(f"\n✓ Saved {len(df_merged)} associations (with original columns) → {args.output}")
    else:
        print("\n⚠ No associations found.")

    # Save cancer+diabetes associations with original columns
    if filtered_rows:
        df_filtered_assoc = pd.DataFrame(filtered_rows)
        
        # Merge with original input data
        df_filtered_merged = df_filtered_assoc.merge(
            df_input,
            left_on='rsid',
            right_on=args.column,
            how='left'
        )
        
        filtered_output = args.output.replace(".csv", "_cancer_diabetes.csv")
        df_filtered_merged.to_csv(filtered_output, index=False)
        print(f"✓ Saved {len(df_filtered_merged)} cancer/diabetes associations (with original columns) → {filtered_output}")
    else:
        print("\n⚠ No cancer or diabetes associations found.")


if __name__ == "__main__":
    main()