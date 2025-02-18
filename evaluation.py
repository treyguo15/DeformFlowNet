import torch
# SR : Segmentation Result
# GT : Ground Truth
def get_accuracy(SR,GT):
    row,col=SR.shape
    TP,FN,FP,TN=0,0,0,0
    for i in range(row):
        for j in range(col):
            if SR[i][j]==255 and GT[i][j]==255:
                TP+=1    
            elif SR[i][j]==0 and GT[i][j]==255:
                FN+=1
            elif SR[i][j]==255 and GT[i][j]==0:
                FP+=1 
            else: 
                TN+=1 
    ACC=(TP+TN)/(TP+FN+TN+FP)
    #print(TP,TN,FN,FP)
    return ACC

def get_sensitivity(SR,GT):
    row,col=SR.shape
    TP,FN=0,0
    for i in range(row):
        for j in range(col):
            if SR[i][j]==255 and GT[i][j]==255:
                TP+=1
            elif SR[i][j]==0 and GT[i][j]==255:
                FN+=1
    SE=TP/(TP+FN)
    return SE

def get_specificity(SR,GT):
    row,col=SR.shape
    TN,FP=0,0
    for i in range(row):
        for j in range(col):
            if SR[i][j]==0 and GT[i][j]==0:
                TN+=1
            elif SR[i][j]==255 and GT[i][j]==0:
                FP+=1
    SP=TN/(TN+FP)
    return SP
 
def get_precision(SR,GT):
    row,col=SR.shape
    TP,FP=0,0
    for i in range(row):
        for j in range(col):
            if SR[i][j]==255 and GT[i][j]==255:
                TP+=1
            elif SR[i][j]==255 and GT[i][j]==0:
                FP+=1
    PC=TP/(TP+FP) 
    return PC
 
def get_F1(SR,GT):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT)
    PC = get_precision(SR,GT) 
    F1 = 2*SE*PC/(SE+PC) 
    return F1
 
def get_JS(SR,GT):
    row,col=SR.shape
    TP,p=0,0
    for i in range(row):
        for j in range(col):
            if SR[i][j]==255 and GT[i][j]==255:
                TP+=1
            if SR[i][j]==255 or GT[i][j]==255:
                p+=1
    JS=TP/p 
    return JS
 
def get_DC(SR,GT):
    row,col=SR.shape
    TP,SUMA,SUMB=0,0,0
    for i in range(row):
        for j in range(col):
            if SR[i][j]==255 and GT[i][j]==255:
                TP+=1
            SUMA+=SR[i][j]
            SUMB+=GT[i][j]
#           elif SR[i][j]==255:
#                SUMA+=1
#            elif GT[i][j]==1:
#                SUMB+=1
    DC=(2*TP)/(SUMA/255+SUMB/255)
    #print(TP,SUMA/255,SUMB)
    return DC