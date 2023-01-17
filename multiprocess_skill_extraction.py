import pandas as pd
import numpy as np
# skill extraction modules
import spacy
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor

# remove warnings
import warnings
warnings.filterwarnings('ignore')

# parallel processing
from multiprocessing import Pool

# init params of skill extractor
nlp = spacy.load("en_core_web_lg")

# init skill extractor
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

def skill_extraction(df):
    skills_df = pd.DataFrame(columns=['skill_id', 'doc_node_value'])
    skills_df_ngram = pd.DataFrame(columns=['skill_id', 'doc_node_value'])
    for i in range(len(df)):
        print("Progress: [{0:50s}] {1:.1f}%".format('#' * int(i / len(df) * 50), i / len(df) * 100), end='\r')
        try:
            annotations = skill_extractor.annotate(df['Descriptions'][i])
            skills_df_sample = pd.DataFrame(annotations['results']['full_matches'], columns=['skill_id', 'doc_node_value'])
            skills_df = skills_df.append(skills_df_sample)
            skills_df_ngram_sample = pd.DataFrame(annotations['results']['ngram_scored'], columns=['skill_id', 'doc_node_value'])
            skills_df_ngram = skills_df.append(skills_df_ngram_sample)
        except:
            pass
    return skills_df, skills_df_ngram

# run function in parallel
if __name__ == '__main__':
    # load data
    df = pd.read_csv('webscraping_results_assignment3.csv')
    pool_size = 12
    # split data into chunks
    full_match_df = pd.DataFrame(columns=['skill_id', 'doc_node_value'])
    ngram_df = pd.DataFrame(columns=['skill_id', 'doc_node_value'])
    df_split = np.array_split(df, pool_size)
    # run function in parallel
    with Pool(pool_size) as p:
        results = p.map(skill_extraction, df_split)
    # combine results
    for result in results:
        full_match_df = full_match_df.append(result[0])
        ngram_df = ngram_df.append(result[1])

    # save results
    full_match_df.to_csv('full_match_df.csv', index=False)
    ngram_df.to_csv('ngram_df.csv', index=False)