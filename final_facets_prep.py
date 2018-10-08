#from utils.storage.managers import storage_for_path
#from utils.fingerprint import compact_fingerprint_path

import pandas as pd
from PIL import Image
import math
import argparse
import boto3
import os
from audit_benchmarks import AuditBenchmark
import urllib
import io

class FacetsPrep:

  """
  Method to go from input csv of benchmark images, run audits and generate url for Facets Dive demo.

  Sample Implementations:

  [ input csv file with url images ]
  python final_facets_prep.py -i /Users/deborahraji/Downloads/AIAW_results.csv -u

  [ input csv file with local file images in specified base directory, to generate results ]
   python final_facets_prep.py -i /Users/deborahraji/Downloads/AIAW_results.csv -r -b '/Users/deborahraji/Downloads/allPPB-Original/'

  """

  def __init__(self,
               csv_file,
               base_dir,
               filter,
               url_input=False,
               get_results=False):

    self.csv_file = csv_file
    self.base_dir = base_dir
    self.url_input = url_input
    self.get_results = get_results
    self.filter = filter
    self.filename = 'audit_results'


  def __crop_image(self, img):

    if img.width > img.height:
        new_width = img.height
        new_height = img.height

        start_x = (img.width - img.height) / 2
        start_y = 0

        img = img.crop((
            start_x,
            start_y,
            start_x + new_width,
            start_y + new_height,
        ))
    elif img.height > img.width:
        new_width = img.width
        new_height = img.width

        start_x = 0
        start_y = (img.height - img.width) / 2

        img = img.crop((
            start_x,
            start_y,
            start_x + new_width,
            start_y + new_height,
        ))

    return img

  def generate_spritemap(self):
    dict_list = self.data
    PIL_concept_images = []

    for line in dict_list:
      if self.url_input:
          fd = urllib.request.urlopen(line['url'])
          img_file = io.BytesIO(fd.read())
      else:
        img_file = self.base_dir + line['filename']

      im = Image.open(img_file)
      PIL_concept_images.append(im)

    im_list = []
    for each in PIL_concept_images:
      each = self.__crop_image(each)
      each.thumbnail((50, 50))
      im_list.append(each)

    width = 50
    height = 50
    rows = int(math.sqrt(len(im_list))) + 1
    columns = int(math.sqrt(len(im_list))) + 1

    # creates a new empty image, RGB mode
    spritemap = Image.new('RGB', (columns * width, rows * height))

    i = 0
    j = 0
    row_counter = 0
    for k in range(len(im_list)):
        print (i, j)
        spritemap.paste(im_list[k], (i, j))
        if row_counter == rows - 1:
            row_counter = 0
            i = 0
            j += height
        else:
            i += width
            row_counter += 1

    spritemap_filename = self.filename+'.png'
    spritemap.save(spritemap_filename)

    return spritemap_filename

  def generate_json(self):

    #generate results first
    if self.get_results:
        audit = AuditBenchmark()
        res_dict = audit.get_results(self.csv_file, self.base_dir, self.filter, self.url_input)
        csv_data = pd.DataFrame.from_dict(res_dict)
    else:
        csv_data = pd.DataFrame.from_csv(self.csv_file)

    data_dicts = list()
    for i in range(csv_data.shape[0]):
      row_dict = dict()
      for item in csv_data:
        row_dict[item] = list(csv_data[item])[i]
      data_dicts.append(row_dict)

    #create a local json file
    json_filename = self.filename + '.json'
    f = open(json_filename, "w")
    f.write(str(data_dicts))

    self.data = data_dicts

    return json_filename

  def generate_url(self):
      #rehost png and json
      s3 = boto3.client('s3')
      bucket_name = 'audit-tool-uploads'

      json_file = self.generate_json()
      spritemap_file = self.generate_spritemap()

      s3.upload_file(json_file, bucket_name, json_file, ExtraArgs={'ACL': 'public-read'})
      s3.upload_file(spritemap_file, bucket_name, spritemap_file, ExtraArgs={'ACL': 'public-read'})

      url_stem = 'https://s3-us-west-2.amazonaws.com/'+bucket_name+'/'+self.filename

      #delete residual files
      os.remove(json_file)
      os.remove(spritemap_file)

      return url_stem



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_file', '-i', required=True, help='input csv file with benchmark data')
    parser.add_argument('--url_input', '-u', action='store_true', help='Flag if url input instead of local file input')
    parser.add_argument('--get_results', '-r', action='store_true', help='Flag if we need to populate csv with results before rehosting')
    parser.add_argument('--base_dir', '-b', default='/Users/deborahraji/Downloads/allPPB-Original/', help='base directory for input files')
    parser.add_argument('--filter', '-f', default='A,K,M,F,I', help='Indicate company APIs to be audited, represented by first letter of company names, seperated by commas. '
                                                                    'Default is "A,K,M,F,I" for all available APIs - Amazon, Kairos, Microsoft, Face++, and IBM. '
                                                                    'Any desired subset of this list is allowed, excluding an empty list.   ')

    args = parser.parse_args()
    #set url flag
    url = False
    if args.url_input:
        url = False

    #initate
    fprep = FacetsPrep(args.input_csv_file, args.base_dir, args.filter, args.url_input, args.get_results)
    url_stem = fprep.generate_url()
    print(url_stem)

