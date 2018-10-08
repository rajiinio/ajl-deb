from flask import Flask, flash, redirect, render_template, request, session, abort
from random import randint
from werkzeug import secure_filename
import pandas
import sys 
import os
sys.path.append(os.path.abspath("/Users/deborahraji/Downloads/play"))
from audit_benchmarks import AuditBenchmark
from audit_calc import  AuditCalculations

app = Flask(__name__)
 
@app.route("/")
def index():
    return "Welcome to AJL Auditor Tool!"
 
@app.route("/viz")
def viz():
    #step 1 : automatically generate json and png files + name them and rehost and replace the url in quickdraw.html
    #step 2 : launch demo via Bazel in different window
    return redirect('http://deborahs-air.lan:6006/facets-dive/demo/quickdraw.html')

@app.route('/upload')
def upload_file():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def uploadr_file():
   if request.method == 'POST':
      f = request.files['file']
      base_dir = request.form['Name'] #Example: '/Users/deborahraji/Downloads/allPPB-Original/'
      model = request.form['submit_button'][0]
      audit = AuditBenchmark()
      data_dict = audit.get_results(os.path.abspath(f.filename), base_dir, model)
      df = pandas.DataFrame.from_dict(data_dict)
      auditcalc = AuditCalculations()
      results = auditcalc.get_results(model, df)
      print(results)
      f.save(secure_filename(f.filename))
      return render_template('table.html', **locals())

if __name__ == "__main__":
    app.run()

