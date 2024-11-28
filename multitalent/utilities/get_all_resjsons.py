import os
import json
import gspread
from batchgenerators.utilities.file_and_folder_operations import *
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
import time
from googleapiclient.errors import HttpError

# Define the scope and credentials for Google Sheets and Drive API
scope = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/drive'
]
creds = ServiceAccountCredentials.from_json_keyfile_name('/home/c306h/Desktop/token/results-426723-4c18ff0bb8bd.json', scope)
client = gspread.authorize(creds)
drive_service = build('drive', 'v3', credentials=creds)

# Function to read JSON files from a directory and its subdirectories
def read_json_files(directory):
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('summary.json'):
                with open(os.path.join(root, file), 'r') as f:
                    json_files.append([json.load(f), join(root, file)])
    return json_files

# Function to create a new spreadsheet in a specific Google Drive folder
def create_spreadsheet_in_folder(spreadsheet_name, folder_id):
    '''try:
        spreadsheet_body = {
            'name': spreadsheet_name,
            'mimeType': 'application/vnd.google-apps.spreadsheet',
            'parents': [folder_id]
        }
        spreadsheet = drive_service.files().create(body=spreadsheet_body, fields='id').execute()
        spreadsheet_id = spreadsheet.get('id')
        print(f"Spreadsheet created with ID: {spreadsheet_id}")
        return spreadsheet_id
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None'''

    query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.spreadsheet' and name='{spreadsheet_name}'"
    response = drive_service.files().list(q=query, fields='files(id, name)').execute()
    files = response.get('files', [])

    if files:
        spreadsheet_id = files[0]['id']
        print(f"Spreadsheet '{spreadsheet_name}' already exists with ID: {spreadsheet_id}")
    else:
        spreadsheet_body = {
            'name': spreadsheet_name,
            'mimeType': 'application/vnd.google-apps.spreadsheet',
            'parents': [folder_id]
        }
        spreadsheet = drive_service.files().create(body=spreadsheet_body, fields='id').execute()
        spreadsheet_id = spreadsheet.get('id')
        print(f"Created new spreadsheet '{spreadsheet_name}' with ID: {spreadsheet_id}")

    return spreadsheet_id

# Function to save data to Google Sheets
def save_to_google_sheets(data, spreadsheet_id, worksheet_name):
    spreadsheet = client.open_by_key(spreadsheet_id)
    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows="100", cols="20")
    worksheet.clear()
    # Write data to the worksheet
    for row in data:
        time.sleep(1)
        worksheet.append_row(row)

# Main function
def main():
    old_d = '/home/c306h/cluster-checkpoints/multitalent/nnUNet_trained_models/'
    directory = '/home/c306h/cluster-checkpoints/multitalent/nnUNet_trained_models/'
    drive_folder_id = '1fhvCm1krq9b7-D2lJ87qic6ZXI4K_twt'  # Replace with your Google Drive folder ID
    metrics = ['Dice']
    id_list = [402, 418, 420, 422] #[201, 204, 205, 206, 207]
    for metric in metrics:
        json_data = read_json_files(directory)
        tasks = {}
        for i in json_data:
            temp_path = i[1][len(old_d):]
            path_splits = temp_path.split('/')
            d_name = path_splits[0]
            print(d_name)
            if d_name == 'maxz':
                continue
            if int(d_name[7:10]) in id_list:
                if d_name not in tasks.keys():
                    tasks[d_name] = []
                    dataset_json = load_json(join(old_d, *path_splits[:-3], 'dataset.json'))
                    classes = []
                    for t in dataset_json['labels'].keys():
                        if t != 'background':
                            classes.append(t + '__' +str(dataset_json['labels'][t]))
                    classes.append(('mean'))
                    tasks[d_name] = [['trainer', 'plan', 'config', 'pretrained', 'fold'] + classes]
                if path_splits[1] != 'pretrained':
                    trainer_name = path_splits[1].split('__')[0]
                    plan = path_splits[1].split('__')[1]
                    config = path_splits[1].split('__')[2]
                    pretrained = 'no'
                    fold = path_splits[2]
                    results = []
                    for c in i[0]['mean'].keys():
                        results.append(i[0]['mean'][c][metric])
                    results.append(i[0]['foreground_mean'][metric])
                    tasks[d_name].append([trainer_name, plan,config, pretrained, fold, *results])

                else:
                    if path_splits[2].startswith('Dataset'):
                        trainer_name = path_splits[3].split('__')[0]
                        plan = path_splits[3].split('__')[1]
                        config = path_splits[3].split('__')[2]
                        pretrained = path_splits[2]
                        fold = path_splits[4]
                        results = []
                        for c in i[0]['mean'].keys():
                            results.append(i[0]['mean'][c][metric])
                        results.append(i[0]['foreground_mean'][metric])
                        tasks[d_name].append([trainer_name, plan,config, pretrained, fold, *results])

        # Save to different spreadsheets in the specified Google Drive folder
        for task in tasks.keys():
            spreadsheet_name = task
            spreadsheet_id = create_spreadsheet_in_folder(spreadsheet_name, drive_folder_id)
            save_to_google_sheets(tasks[task], spreadsheet_id, metric)
            print(task)

if __name__ == '__main__':
    main()