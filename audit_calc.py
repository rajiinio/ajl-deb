import pandas
import argparse
import collections
import matplotlib.pyplot as plt
import numpy as np

class AuditCalculations:

    def __init__(self, audit_file=None):
        self.file = audit_file
        self.res = collections.defaultdict(list)
        self.box = collections.defaultdict(list)
        if self.file:
            self.df = pandas.read_csv(self.file, dtype='str')

    def get_results(self, model, df):

        """
        Execute calculations from data structure of audit results to get comparative evaluation metrics of model performance.

        Generates a dictionary with the following figures for each corporate black-box API model:
             - N (number of total images)
             - NA (number of faces not processed)
             - TRUE (number of faces correctly gendered)
             - FALSE (number of faces falsely gendered)
             - Accuracy on Processed Faces
             - Error on Processed Faces
             - Overall Error (including faces not processed or misgendered)

        """

        model_dict = {
            "A": "Amazon",
            "K": "Kairos",
            "M": "Microsoft",
            "F": "Face++",
            "I": "IBM"
        }
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
        key = 'Classifier ' + model_dict[model]
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
            if max(tmp) != 0:  #check if no elements in category
                self.res[cat].append(float('%1.11f' % ((tmp[3]/(tmp[0]-tmp[1]))*100)))
                self.res[cat].append(float('%1.11f' % (100 - self.res[cat][4])))
                self.res[cat].append(float('%1.11f' % (((tmp[1]+tmp[2]) / tmp[0])*100)))
            else:
                continue

        return self.res

    def generate_box_plot(self, model, df):

        """
         Generate box plot of distribution of confidence scores of audit predictions for a given corporate black box model.

        """
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_file', '-i', required=True, help='input csv file with benchmark data')
    parser.add_argument('--model', '-m', required=True, help='First letter of black box model used for audit')
    args = parser.parse_args()

    audit = AuditCalculations(args.input_csv_file)
    audit.get_results(args.model, audit.df)
    audit.generate_box_plot(args.model, audit.df)

