from huggingface_hub import login
from datasets import load_dataset
from dotenv import load_dotenv
import pandas as pd
import os
from datasets.utils.logging import disable_progress_bar, enable_progress_bar
from tqdm import tqdm
from Levenshtein import distance


donor_mapping = pd.read_csv("./data/iati_donor_mapping.csv")
global DONOR_MAPPING_DICT
DONOR_MAPPING_DICT = donor_mapping.set_index('Donor code')['reporting_org_ref'].to_dict()


global IATI
global IATI_ORG_DICT
IATI = load_dataset('alex-miller/iati-policy-markers', split='train')
IATI = IATI.filter(lambda example: example['reporting_org_ref'] in DONOR_MAPPING_DICT.values())
IATI_ORG_DICT = dict()

print("Pre-filtering IATI data by org ref...")
disable_progress_bar()
for reporting_org_ref in tqdm(DONOR_MAPPING_DICT.values()):
    IATI_ORG_DICT[reporting_org_ref] = IATI.filter(
        lambda example: example["reporting_org_ref"] == reporting_org_ref, num_proc=8)
enable_progress_bar()

global TEXT_DISTANCE
TEXT_DISTANCE = 3


def substring_matches(short_substring, long_strings):
    matches = []
    if short_substring is None:
        return matches
    for i, long_string in enumerate(long_strings):
        if long_string is not None:
            if short_substring in long_string:
                matches.append(i)
    return matches


def distance_matches(short_substring, long_strings):
    matches = list()
    for i, long_string in enumerate(long_strings):
        if distance(short_substring, long_string) <= TEXT_DISTANCE:
            matches.append(i)
    return matches


# Try to match CRS to IATI by:
# 1. Reporting org ref
# 2. Project Number
# 3. Project Title
# 4. Short Description
# 5. Long Description
def match_crs_iati(crs_example):
    crs_example["matched_iati_identifier"] = ""
    crs_example["iati_text"] = ""
    donor_code = crs_example["DonorCode"]
    try:
        # Reporting org ref
        reporting_org_ref = DONOR_MAPPING_DICT[donor_code]
        org_iati = IATI_ORG_DICT[reporting_org_ref]


        # Project number
        project_number = crs_example["ProjectNumber"]
        if project_number is not None:
            theoretical_iati_identifier = "{}-{}".format(reporting_org_ref, project_number)
            projectnumber_match_indices = distance_matches(theoretical_iati_identifier, org_iati["iati_identifier"])
            if len(projectnumber_match_indices) > 0:
                iati_match = org_iati.select(projectnumber_match_indices)
                crs_example["matched_iati_identifier"] = iati_match["iati_identifier"][0]
                crs_example["iati_text"] = iati_match["text"][0]
                return crs_example

        # Project title
        project_title = crs_example["ProjectTitle"]
        if project_title is not None and len(project_title) > 10:
            projecttitle_match_indices = substring_matches(project_title, org_iati["text"])
            if len(projecttitle_match_indices) > 0:
                iati_match = org_iati.select(projecttitle_match_indices)
                crs_example["matched_iati_identifier"] = iati_match["iati_identifier"][0]
                crs_example["iati_text"] = iati_match["text"][0]
                return crs_example

        # Short description
        short_description = crs_example["ShortDescription"]
        if short_description is not None and len(short_description) > 10:
            shortdescription_match_indices = substring_matches(short_description, org_iati["text"])
            if len(shortdescription_match_indices) > 0:
                iati_match = org_iati.select(shortdescription_match_indices)
                crs_example["matched_iati_identifier"] = iati_match["iati_identifier"][0]
                crs_example["iati_text"] = iati_match["text"][0]
                return crs_example

        # Long description
        long_description = crs_example["LongDescription"]
        if long_description is not None and len(long_description) > 10:
            longdescription_match_indices = substring_matches(long_description, org_iati["text"])
            if len(longdescription_match_indices) > 0:
                iati_match = org_iati.select(longdescription_match_indices)
                crs_example["matched_iati_identifier"] = iati_match["iati_identifier"][0]
                crs_example["iati_text"] = iati_match["text"][0]
                return crs_example

        return crs_example

    except KeyError:
        return crs_example



def main():
    crs = load_dataset('alex-miller/oecd-dac-crs', split='train')
    crs = crs.filter(lambda example: example["Year"] >= 2017)
    crs = crs.map(match_crs_iati, num_proc=8)

    # Push
    crs.push_to_hub("alex-miller/iati-text-enhanced-crs")


if __name__ == '__main__':
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    login(token=HF_TOKEN)
    main()