import pandas as pd 
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np


openClassGeminiPerformance = [1 ,0, 0 ,1 ,1, 0, 1, 1, 1 ,0, 0, 0 ,1, 0, 1, 1 ,1, 1, 1 ,1, 0 ,0]
elevenClassGeminiPerformance = [1, 0 ,0 ,1 ,1, 1, 1 ,1, 1, 0 ,1, 0 ,1 ,0 ,1 ,1 ,1, 1, 1, 1, 0, 1]

openClassGPTPerformance = [1,0,1,1,1,1,1,1,1,0,0,0,1,0,0,1,1,0,1,1,0,0]
elevenClassGPTPerformance = [1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0]

## Analysis of correlation between availability and LLM performance via Chi-Square-Test 

onlineAvailability = [1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

print('Analysis of correlation: online availability ~ performance LLMs')

def chi2_contingency_test(data, target):
    contingency_table = pd.crosstab(data, target)
    results = chi2_contingency(contingency_table)
    return results

results_openClass_gemini_availability = chi2_contingency_test(onlineAvailability, openClassGeminiPerformance)
print("p Wert für Korrelationstest OpenClass Gemini Availability: ", results_openClass_gemini_availability[1])

results_elevenClass_gemini_availability = chi2_contingency_test(onlineAvailability, elevenClassGeminiPerformance)
print("p Wert für Korrelationstest ElevenClass Gemini Availability: ", results_elevenClass_gemini_availability[1])

results_openClass_gpt_availability = chi2_contingency_test(onlineAvailability, openClassGPTPerformance)
print("p Wert für Korrelationstest OpenClass GPT Availability: ", results_openClass_gpt_availability[1])

results_elevenClass_gpt_availability = chi2_contingency_test(onlineAvailability, elevenClassGPTPerformance)
print("p Wert für Korrelationstest ElevenClass GPT Availability: ", results_elevenClass_gpt_availability[1])


## Analysis of correlation between publication year (considered as continuous variable) and LLM performance via a Logistic Regression an a Chi-Square-Test

print('Analysis of correlation: Publication Year ~ Performance LLMs')
publYear = [1813,1916,2007,1847,1814,1903,1896,1946,2005,1888,2011,2007,1988,1891,1973,1977,2001,1704,1726,1847,1889,1952]
performance_results = [openClassGeminiPerformance, elevenClassGeminiPerformance, openClassGPTPerformance, elevenClassGPTPerformance]

for performance_result in performance_results:
    # Daten in ein Modellformat bringen
    X = np.array(publYear)  # unabhängige Variable (Publikationsjahr)
    y = np.array(performance_result)  # abhängige Variable (Testergebnis)

    # Hinzufügen einer Konstanten für das Interzept
    X_with_const = sm.add_constant(X)

    # Logistische Regression
    logit_model = sm.Logit(y, X_with_const)
    result = logit_model.fit()

    # print results
    #print(result.summary())

    ## Do chi test for groups of years

    # create groups for publication year
    bins = [1700, 1800, 1900, 2000, 2100]
    categories = pd.cut(publYear, bins=bins, labels=["1700-1800", "1800-1900", "1900-2000", "2000-2100"])

    # create cross table
    contingency_table = pd.crosstab(categories, performance_result)

    # Chi-Square-Test
    result = chi2_contingency_test(categories, performance_result)
    print(f"Correlation to Publication Year p-value = {result[1]}")