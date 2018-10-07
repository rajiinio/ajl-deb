import csv
import pandas
import argparse
import collections
import matplotlib.pyplot as plt
import numpy as np

INDEX_REF ={
    'N':0,
    'NA':1,
    'TRUE':2,
    'FALSE':3,
    'Accuracy':4,
    'Error on Processed Faces':5,
    'Overall Error':6
}

class AuditCalculations:

    def __init__(self, audit_file=None):
        self.file = audit_file
        self.res = collections.defaultdict(list)
        self.box = collections.defaultdict(list)
        if self.file:
            self.df = pandas.read_csv(self.file, dtype='str')

    def get_results(self, model, df):
        #Total count
        self.res['All'].append(df.shape[0])
        self.res['Females'].append(df.loc[df['Gender'] == 'Female'].shape[0])
        self.res['Males'].append(df.loc[df['Gender'] == 'Male'].shape[0])
        self.res['Lighter'].append(df.loc[df['Binary Fitzpatrick'] == 'lighter'].shape[0])
        self.res['Darker'].append(df.loc[df['Binary Fitzpatrick'] == 'darker'].shape[0])
        self.res['DF'].append(df.loc[(df['Gender'] == 'Female') & (df['Binary Fitzpatrick'] == 'darker')].shape[0])
        self.res['LF'].append(df.loc[(df['Gender'] == 'Female') & (df['Binary Fitzpatrick'] == 'lighter')].shape[0])
        self.res['DM'].append(df.loc[(df['Gender'] == 'Male') & (df['Binary Fitzpatrick'] == 'darker')].shape[0])
        self.res['LM'].append(df.loc[(df['Gender'] == 'Male') & (df['Binary Fitzpatrick'] == 'lighter')].shape[0])

        # grab NA (Face Not Processed)
        # grab FALSE (Misgendered)
        # grab TRUE
        key = 'Classifier ' + model
        for status in ['NA', 'FALSE', 'TRUE']:
            self.res['All'].append(df.loc[df[key] == status].shape[0])
            self.res['Females'].append(df.loc[(df['Gender'] == 'Female') & (df[key] == status)].shape[0])
            self.res['Males'].append(df.loc[(df['Gender'] == 'Male') & (df[key] == status)].shape[0])
            self.res['Lighter'].append(df.loc[(df['Binary Fitzpatrick'] == 'lighter') & (df[key] == status)].shape[0])
            self.res['Darker'].append(df.loc[(df['Binary Fitzpatrick'] == 'darker') & (df[key] == status)].shape[0])
            self.res['DF'].append(df.loc[(df['Gender'] == 'Female') & (df['Binary Fitzpatrick'] == 'darker') & (df[key] == status)].shape[0])
            self.res['LF'].append(df.loc[(df['Gender'] == 'Female') & (df['Binary Fitzpatrick'] == 'lighter') & (df[key] == status)].shape[0])
            self.res['DM'].append(df.loc[(df['Gender'] == 'Male') & (df['Binary Fitzpatrick'] == 'darker') & (df[key] == status)].shape[0])
            self.res['LM'].append(df.loc[(df['Gender'] == 'Male') & (df['Binary Fitzpatrick'] == 'lighter') & (df[key] == status)].shape[0])


        #Accuracy
        for cat in self.res.keys():
            tmp = self.res[cat]

            self.res[cat].append(float('%1.11f' % ((tmp[3]/(tmp[0]-tmp[1]))*100)))
            self.res[cat].append(float('%1.11f' % (100 - self.res[cat][4])))
            self.res[cat].append(float('%1.11f' % (((tmp[1]+tmp[2]) / tmp[0])*100)))

        return self.res

    def generate_box_plot(self, model, df):
        key = model+' Confidence'
        self.box['All'] = df[key].values.astype(np.float)
        self.box['Females'] = df.loc[df['Gender'] == 'Female'][key].values.astype(np.float)
        self.box['Males'] = df.loc[df['Gender'] == 'Male'][key].values.astype(np.float)
        self.box['Lighter'] = df.loc[df['Binary Fitzpatrick'] == 'lighter'][key].values.astype(np.float)
        self.box['Darker'] = df.loc[df['Binary Fitzpatrick'] == 'darker'][key].values.astype(np.float)
        self.box['DF'] = df.loc[(df['Gender'] == 'Female') & (df['Binary Fitzpatrick'] == 'darker')][key].values.astype(np.float)
        self.box['LF'] = df.loc[(df['Gender'] == 'Female') & (df['Binary Fitzpatrick'] == 'lighter')][key].values.astype(np.float)
        self.box['DM'] = df.loc[(df['Gender'] == 'Male') & (df['Binary Fitzpatrick'] == 'darker')][key].values.astype(np.float)
        self.box['LM'] = df.loc[(df['Gender'] == 'Male') & (df['Binary Fitzpatrick'] == 'lighter')][key].values.astype(np.float)

        #basic analysis
        fig1, axes1 = plt.subplots()
        labels = ['Total', 'Female', 'Male', 'Darker', 'Lighter']
        axes1.boxplot([self.box['All'], self.box['Females'], self.box['Males'], self.box['Darker'], self.box['Lighter']], 0, "", labels=labels)
        axes1.set_title('%s Confidence Distribution (With Outliers)' % model, fontsize=10)

        #intersectional analysis
        fig2, axes2 = plt.subplots()
        labels = ['Lighter Male', 'Lighter Female', 'Darker Male', 'Darker Female']
        axes2.boxplot([self.box['LM'], self.box['LF'], self.box['DM'], self.box['DF']], 0, "", labels=labels)
        axes2.set_title('%s Confidence Distribution (With Outliers)' % model, fontsize=10)

        plt.show()


        return



"""

reader = csv.DictReader(open(ppb_file, 'r'))

dict_list=[]
lm =[]
lf =[]
dm =[]
df =[]
T = []

for line in reader:
    img_file = line['filename']
    gender = img_file.split('_')[2]
    #a_gen = line['Amazon Results']
    a_gen = line['Amazon Confidence']

    #males
    if line['Binary Fitzpatrick'] == 'lighter':
        i = 0
    else:
        i = 1

    T.append(a_gen)
    if (gender == 'm') and i == 0:
        lm.append(a_gen)

    if (gender == 'm') and i == 1:
        dm.append(a_gen)

    if (gender == 'f') and i == 0:
        lf.append(a_gen)

    if (gender == 'f') and i == 1:
        df.append(a_gen)

    
    if (a_gen == 'm') and (gender == 'm'):
        TP_M[i] += 1
        TN_F[i] += 1

    elif (a_gen == 'f') and (gender == 'f'):
        TP_F[i] += 1
        TN_M[i] += 1

    elif (a_gen == 'm') and (gender == 'f'):
        FP_M[i] += 1
        FN_F[i] += 1

    elif (a_gen == 'f') and (gender == 'm'):
        FP_F[i] += 1
        FN_M[i] += 1
    
#print(lm,lf,dm,df, T)

import matplotlib.pyplot as plt
import numpy as np

spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low), 0)
T = np.array(T).astype(np.float)
lf = np.array(lf).astype(np.float)
lm = np.array(lm).astype(np.float)
df = np.array(df).astype(np.float)
dm = np.array(dm).astype(np.float)

fig, axes = plt.subplots()
#print(data.shape)
#print(T)
"""
"""
plt.figure()
labels =['Lighter Male', 'Lighter Female', 'Darker Male', 'Darker Female']
axes.boxplot([lm, lf, dm, df], 0, "", labels=labels)
#axes.boxplot(data)
axes.set_title('Amazon Confidence Distribution (With Outliers)', fontsize=10)


plt.figure()
axes.boxplot([T],labels=['All Demographics'])
#axes.boxplot(data)
axes.set_title('Amazon Confidence Distribution (With Outliers)', fontsize=10)
#plt.show()



plt.figure()
df = pandas.read_csv(ppb_file)
ax = df.boxplot(column='Amazon Confidence', by='Fitzpatrick Skin Type')
#ax.set_title('Amazon Confidence Distribution by Binary Fitzpatrick (With Outliers)', fontsize=10)

plt.show()


print ("TP_M", TP_M)
print ("TP_F", TP_F)
print ("TN_M", TN_M)
print ("TN_F", TN_F)

print ("FP_M", FP_M)
print ("FP_F", FP_F)
print ("FN_M", FN_M)
print ("FN_F", FN_F)





    #k_gen = k_detect_faces(filename)
    #line['Amazon Results'] = a_gen


#generate results
#PPV_LM = (TP_M[0]+ TN_M[0])/(TP_M[0] + FP_M[0] + FN_M[0] + TN_M[0])
PPV_LM = TP_M[0]/(TP_M[0] + FP_M[0])
TPR_LM = TP_M[0]/(TP_M[0] + FN_M[0])
FPR_LM = FP_M[0]/(FP_M[0] + TN_M[0])
ER_LM = 1 - PPV_LM
ER_II_LM = 1 - TPR_LM

#PPV_DM = (TP_M[1]+ TN_M[1])/(TP_M[1] + FP_M[1] + FN_M[1] + TN_M[1])
PPV_DM = TP_M[1]/(TP_M[1] + FP_M[1])
TPR_DM = TP_M[1]/(TP_M[1] + FN_M[1])
FPR_DM = FP_M[1]/(FP_M[1] + TN_M[1])
ER_DM = 1 - PPV_DM
ER_II_DM = 1 - TPR_DM

PPV_M = sum(TP_M)/(sum(TP_M) + sum(FP_M))
#PPV_M = (sum(TP_M)+ sum(TN_M))/(sum(TP_M) + sum(FP_M) + sum(FN_M) + sum(TN_M))
TPR_M = sum(TP_M)/(sum(TP_M)+ sum(FN_M))
FPR_M = sum(FP_M)/(sum(FP_M) + sum(TN_M))
ER_M = 1 - PPV_M
ER_II_M = 1 - TPR_M

print (PPV_LM, PPV_DM, PPV_M)
print (TPR_LM, TPR_DM, TPR_M)
print (FPR_LM, FPR_DM, FPR_M)
print (ER_LM, ER_DM, ER_M)
print (ER_II_LM, ER_II_DM, ER_II_M)

PPV_LF = TP_F[0]/(TP_F[0] + FP_F[0])
#PPV_LF = (TP_F[0]+ TN_F[0])/(TP_F[0] + FP_F[0] + FN_F[0] + TN_F[0])
TPR_LF = TP_F[0]/(TP_F[0] + FN_F[0])
FPR_LF = FP_F[0]/(FP_F[0] + TN_F[0])
ER_LF = 1 - PPV_LF
ER_II_LF = 1 - TPR_LF

PPV_DF = TP_F[1]/(TP_F[1] + FP_F[1])
#PPV_DF = (TP_F[1]+ TN_F[1])/(TP_F[1] + FP_F[1] + FN_F[1] + TN_F[1])
TPR_DF = TP_F[1]/(TP_F[1] + FN_F[1])
FPR_DF = FP_F[1]/(FP_F[1] + TN_F[1])
ER_DF = 1 - PPV_DF
ER_II_DF = 1 - TPR_DF

PPV_F = sum(TP_F)/(sum(TP_F) + sum(FP_F))
#PPV_F = (sum(TP_F)+ sum(TN_F))/(sum(TP_F) + sum(FP_F) + sum(FN_F) + sum(TN_F))
TPR_F = sum(TP_F)/(sum(TP_F)+ sum(FN_F))
FPR_F = sum(FP_F)/(sum(FP_F) + sum(TN_F))
ER_F = 1 - PPV_F #type 1
ER_II_F = 1 - TPR_F

#recall = TPR
#precision = PPV
print ('**************')
print (PPV_LF, PPV_DF, PPV_F)
print (TPR_LF, TPR_DF, TPR_F)
print (FPR_LF, FPR_DF, FPR_F)
print (ER_LF, ER_DF, ER_F)
print (ER_II_LF, ER_II_DF, ER_II_F)

"""

#male, female
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_file', '-i', required=True, help='input csv file with benchmark data')
    parser.add_argument('--model', '-m', required=True, help='black box model used for audit')
    args = parser.parse_args()

    #ppb_file = 'PPB_extended_x.csv'
    audit = AuditCalculations(args.input_csv_file)
    audit.get_results(args.model, audit.df)
    audit.generate_box_plot(args.model, audit.df)

