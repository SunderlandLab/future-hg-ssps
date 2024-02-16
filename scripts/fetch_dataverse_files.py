import requests
import os.path
import json

print('STARTING SCRIPT: fetch_dataverse_files.py')

output_path = './'

# read Dataverse repository metadata
with open('./data/dataverse_metadata.json') as f:
    metadata = json.load(f)

# loop through files and download
for i in metadata['datasetVersion']['files']:
    id = i['dataFile']['id']
    fn = i['dataFile']['filename']
    dir = i['directoryLabel']
    
    out_path = f'{output_path}/{dir}/{fn}'

    # delete file if it exists:
    check_file = os.path.exists(out_path)
    if check_file == True:
        os.remove(out_path)
    # download file
    r = requests.get(f'https://dataverse.harvard.edu/api/access/datafile/{id}', stream=True)
    print(out_path)
    with open(out_path, 'wb') as outfile:
        #print(f'Downloading {fn}...')
        outfile.write(r.content)

print('COMPLETED SCRIPT: fetch_dataverse_files.py')