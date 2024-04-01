from huggingface_hub import login
from datasets import load_dataset
from dotenv import load_dotenv
import pandas as pd
import os
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
from tqdm import tqdm


donor_mapping = pd.read_csv("./data/iati_donor_mapping.csv")
global DONOR_MAPPING_DICT
DONOR_MAPPING_DICT = donor_mapping.set_index('Donor code')['reporting_org_ref'].to_dict()


def map_activity_dates(example):
    min_year = 9999
    max_year = 0
    activity_dates = example["activity_dates"].split("|")
    for activity_date in activity_dates:
        try:
            year = int(activity_date[:4])
            if year > max_year:
                max_year = year
            if year < min_year:
                min_year = year
        except ValueError:
            pass
    example["min_year"] = min_year
    example["max_year"] = max_year
    return example


global IATI
global IATI_ORG_DICT
IATI = load_dataset('alex-miller/iati-policy-markers', split='train')
print("Splitting IATI dates...")
IATI = IATI.map(map_activity_dates, num_proc=8, remove_columns=["activity_dates"])
print("Pre-filtering IATI data...")
IATI = IATI.filter(lambda example:
                    example['reporting_org_ref'] in DONOR_MAPPING_DICT.values() and
                    example['max_year'] >= 2017
)
cols_to_remove = IATI.column_names
cols_to_remove.remove("reporting_org_ref")
cols_to_remove.remove("min_year")
cols_to_remove.remove("max_year")
cols_to_remove.remove("iati_identifier")
cols_to_remove.remove("text")
IATI = IATI.remove_columns(cols_to_remove)
IATI_ORG_DICT = dict()


print("Pre-sorting IATI data by org ref...")
disable_progress_bar()
for reporting_org_ref in tqdm(DONOR_MAPPING_DICT.values()):
    IATI_ORG_DICT[reporting_org_ref] = IATI.filter(
        lambda example: example["reporting_org_ref"] == reporting_org_ref, num_proc=8)
enable_progress_bar()


def substring_matches(short_substring, long_strings):
    matches = []
    if short_substring is None:
        return matches
    for i, long_string in enumerate(long_strings):
        if long_string is not None:
            if short_substring.lower() in long_string.lower():
                matches.append(i)
    return matches


def temporal_matches(crs_year, iati_min_years, iati_max_years):
    matches = []
    for i, min_year in enumerate(iati_min_years):
        max_year = iati_max_years[i]
        if crs_year >= min_year and crs_year <= max_year:
            matches.append(i)
    return matches


# Try to match CRS to IATI by:
# 1. Reporting org ref
# 2. Years
# 3. Project Number
# 4. Project Title
# 5. Short Description
# 6. Long Description
# No more than 5 matching activities are considered, because any more means
# it's likely to be a meaningless match, but some might happen
def match_crs_iati(crs_example):
    crs_example["matched_iati_identifier"] = ""
    crs_example["iati_text"] = ""
    crs_example["iati_match_type"] = ""
    donor_code = crs_example["DonorCode"]
    try:
        # Reporting org ref
        reporting_org_ref = DONOR_MAPPING_DICT[donor_code]
        org_iati = IATI_ORG_DICT[reporting_org_ref]

        # Year match
        crs_year = crs_example["Year"]
        year_match_indicies = temporal_matches(crs_year, org_iati["min_year"], org_iati["max_year"])
        if len(year_match_indicies) == 0:
            return crs_example
        org_iati = org_iati.select(year_match_indicies)

        # Project number
        project_number = crs_example["ProjectNumber"]
        if project_number is not None:
            theoretical_iati_identifier = "{}-{}".format(reporting_org_ref, project_number)
            projectnumber_match_indices = substring_matches(theoretical_iati_identifier, org_iati["iati_identifier"])
            if len(projectnumber_match_indices) > 0 and len(projectnumber_match_indices) <= 5:
                iati_match = org_iati.select(projectnumber_match_indices)
                crs_example["matched_iati_identifier"] = iati_match["iati_identifier"][0]
                crs_example["iati_text"] = iati_match["text"][0]
                crs_example["iati_match_type"] = "identifier"
                return crs_example

        # Project title
        project_title = crs_example["ProjectTitle"]
        if project_title is not None and len(project_title) > 20:
            projecttitle_match_indices = substring_matches(project_title, org_iati["text"])
            if len(projecttitle_match_indices) > 0 and len(projecttitle_match_indices) <= 5:
                iati_match = org_iati.select(projecttitle_match_indices)
                crs_example["matched_iati_identifier"] = iati_match["iati_identifier"][0]
                crs_example["iati_text"] = iati_match["text"][0]
                crs_example["iati_match_type"] = "project title"
                return crs_example

        # Short description
        short_description = crs_example["ShortDescription"]
        if short_description is not None and len(short_description) > 20:
            shortdescription_match_indices = substring_matches(short_description, org_iati["text"])
            if len(shortdescription_match_indices) > 0 and len(shortdescription_match_indices) <= 5:
                iati_match = org_iati.select(shortdescription_match_indices)
                crs_example["matched_iati_identifier"] = iati_match["iati_identifier"][0]
                crs_example["iati_text"] = iati_match["text"][0]
                crs_example["iati_match_type"] = "short description"
                return crs_example

        # Long description
        long_description = crs_example["LongDescription"]
        if long_description is not None and len(long_description) > 20:
            longdescription_match_indices = substring_matches(long_description, org_iati["text"])
            if len(longdescription_match_indices) > 0 and len(longdescription_match_indices) <= 5:
                iati_match = org_iati.select(longdescription_match_indices)
                crs_example["matched_iati_identifier"] = iati_match["iati_identifier"][0]
                crs_example["iati_text"] = iati_match["text"][0]
                crs_example["iati_match_type"] = "long description"
                return crs_example

        return crs_example

    except KeyError:
        return crs_example


def main():
    crs = load_dataset('alex-miller/oecd-dac-crs', split='train')
    crs = crs.filter(lambda example: example["Year"] >= 2017)
    # crs = crs.shuffle(seed=1337).select(range(10000))
    crs = crs.map(match_crs_iati, num_proc=8)

    # Push
    crs.push_to_hub("alex-miller/iati-text-enhanced-crs")
    # crs.to_csv("./large_data/iati-text-enhanced-crs.csv")


if __name__ == '__main__':
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    login(token=HF_TOKEN)
    main()