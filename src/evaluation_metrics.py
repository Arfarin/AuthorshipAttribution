import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# testset tatsächliche labels
y_true_2 = [1, 2, 3, 4, 1, 5, 6, 7, 3, 2, 8, 8, 9, 5, 9, 10, 10, 11, 11, 4, 6, 7 ]

def resultsGPTfileInput():
    y_pred_csvOhneAbstand= [0,0,0,0,5,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
    y_pred_csvMitAbstand = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,11,4,6,7]
    
    # GPT csvOhneAbstand
    print("GPT: input via csv file ohne Leerzeilen")
    f1_macro_gpt_csvOA = f1_score(y_true_2, y_pred_csvOhneAbstand, average='macro')
    print("F1 Macro GPt: ", f1_macro_gpt_csvOA)
    f1_micro_gpt_csvOA = f1_score(y_true_2, y_pred_csvOhneAbstand, average='micro')
    print("F1 Micro GPt: ", f1_micro_gpt_csvOA)
    accuracy_gpt_csvOA = accuracy_score(y_true_2, y_pred_csvOhneAbstand)
    print("Accuracy GPT: ", accuracy_gpt_csvOA)
    
    # GPT csvMitAbstand
    print("GPT:  input via csv file mit Leerzeilen")
    f1_macro_gpt_csvMA = f1_score(y_true_2, y_pred_csvMitAbstand, average='macro')
    print("F1 Macro GPt: ", f1_macro_gpt_csvMA)
    f1_micro_gpt_csvMA = f1_score(y_true_2, y_pred_csvMitAbstand, average='micro')
    print("F1 Micro GPt: ", f1_micro_gpt_csvMA)
    accuracy_gpt_csvMA = accuracy_score(y_true_2, y_pred_csvMitAbstand)
    print("Accuracy GPT: ", accuracy_gpt_csvMA)


## open class setting 
def openClassResults():
    
    print("Open class setting")
    
    # author id 2 used
    # class for undefined authors gets id 0
    y_pred_gpt_2 = [1,0,3,4,1,5,6,7,3,0,0,0,9,6,0,10,10,0,11,4,0,0]
    y_pred_gem_2 = [1,0,0,4,1,0,6,7,3,1,0,1,9,0,9,10,10,11,11,4,0,0]
    
    ## second IDs
    print("")
    # GPT
    print("second IDs")
    print("")
    print("GPT")
    f1_macro_gpt_open = f1_score(y_true_2, y_pred_gpt_2, average='macro')
    print("F1 Macro GPt: ", f1_macro_gpt_open)
    f1_micro_gpt_open = f1_score(y_true_2, y_pred_gpt_2, average='micro')
    print("F1 Micro GPt: ", f1_micro_gpt_open)
    accuracy_gpt_open = accuracy_score(y_true_2, y_pred_gpt_2)
    print("Accuracy GPT: ", accuracy_gpt_open)

    print("")
    print("Gemini:")
    # Gemini
    f1_macro_gem_open = f1_score(y_true_2, y_pred_gem_2, average='macro')
    print("F1 Macro Gemini: ", f1_macro_gem_open)
    f1_micro_gem_open = f1_score(y_true_2, y_pred_gem_2, average='micro')
    print("F1 Micro Gemini: ", f1_micro_gem_open)
    accuracy_gem_open = accuracy_score(y_true_2, y_pred_gem_2)
    print("Accuracy Gemini: ", accuracy_gem_open)




## 11 Class setting

def elevenClassResults():

    print("11 Class setting")

    y_true =     [26, 23, 4, 14, 26, 9, 8, 5, 4, 23, 6, 6, 2, 9, 2, 27, 27, 7, 7, 14, 8, 5 ]
    y_pred_gpt = [26, 8, 4, 14, 26, 9, 8, 5, 4, 23, 6, 14, 2, 8, 2, 27, 27, 8, 7, 14, 8, 27 ]
   # y_pred_gem = [26, 0, 4, 14, 26, 9, 8, 5, 4, 26, 6, 26, 2, 0, 2, 27, 27, 7, 7, 14, 9, 5 ]
    y_pred_gem = [26,0,27,14,26,9,8,5,4,26,6,26,2,0,2,27,27,7,7,14,9,5]

    # author id 2 used
    #y_true_2 = [1, 2, 3, 4, 1, 5, 6, 7, 3, 2, 8, 8, 9, 5, 9, 10, 10, 11, 11, 4, 6, 7 ]
    y_pred_gpt_2 = [1, 6, 3, 4, 1, 5 ,6, 7, 3, 2, 8, 4, 9, 6, 9, 10, 10, 6, 11, 4, 6, 10 ]
    #y_pred_gem_2 = [1, 0, 3, 4, 1, 5, 6, 7, 3, 1, 8, 1, 9, 0, 9, 10, 10 ,11, 11, 4, 5, 7 ]
    y_pred_gem_2 =[1,0,10,4,1,5,6,7,3,1,8,1,9,0,9,10,10,11,11,4,5,7]

    # GPT second trial 11 class autho id 1 
    y_pred_gpt_secondTrial = [ 26, 23, 4, 14, 26, 9, 8, 5, 4, 23, 6, 14, 2, 9, 4, 27, 27, 8, 7, 14, 23, 5 ]

    # GPT
    f1_macro_gpt_11 = f1_score(y_true, y_pred_gpt, average='macro')
    print("F1 Macro GPt: ", f1_macro_gpt_11)
    f1_micro_gpt_11 = f1_score(y_true, y_pred_gpt, average='micro')
    print("F1 Micro GPt: ", f1_micro_gpt_11)
    accuracy_gpt_11 = accuracy_score(y_true, y_pred_gpt)
    print("Accuracy GPT: ", accuracy_gpt_11)

    print("second trial GPT") 
    f1_macro_gpt_11 = f1_score(y_true, y_pred_gpt_secondTrial, average='macro')
    print("F1 Macro GPt: ", f1_macro_gpt_11)
    f1_micro_gpt_11 = f1_score(y_true, y_pred_gpt_secondTrial, average='micro')
    print("F1 Micro GPt: ", f1_micro_gpt_11)
    accuracy_gpt_11 = accuracy_score(y_true, y_pred_gpt_secondTrial)
    print("Accuracy GPT: ", accuracy_gpt_11)


    print("")
    print("Gemini")
    # Gemini
    f1_macro_gem_11 = f1_score(y_true, y_pred_gem, average='macro')
    print("F1 Macro Gemini: ", f1_macro_gem_11)
    f1_micro_gem_11 = f1_score(y_true, y_pred_gem, average='micro')
    print("F1 Micro Gemini: ", f1_micro_gem_11)
    accuracy_gem_11 = accuracy_score(y_true, y_pred_gem)
    print("Accuracy Gemini: ", accuracy_gem_11)

    ## second IDs
    # GPT
    print("")
    print("second IDs")
    print("")
    print("GPT")
    f1_macro_gpt_11 = f1_score(y_true_2, y_pred_gpt_2, average='macro')
    print("F1 Macro GPt: ", f1_macro_gpt_11)
    f1_micro_gpt_11 = f1_score(y_true_2, y_pred_gpt_2, average='micro')
    print("F1 Micro GPt: ", f1_micro_gpt_11)
    accuracy_gpt_11 = accuracy_score(y_true_2, y_pred_gpt_2)
    print("Accuracy GPT: ", accuracy_gpt_11)

    print("")
    print("Gemini:")
    # Gemini
    f1_macro_gem_11 = f1_score(y_true_2, y_pred_gem_2, average='macro')
    print("F1 Macro Gemini: ", f1_macro_gem_11)
    f1_micro_gem_11 = f1_score(y_true_2, y_pred_gem_2, average='micro')
    print("F1 Micro Gemini: ", f1_micro_gem_11)
    accuracy_gem_11 = accuracy_score(y_true_2, y_pred_gem_2)
    print("Accuracy Gemini: ", accuracy_gem_11)
    
    
resultsGPTfileInput()
openClassResults()    
elevenClassResults()