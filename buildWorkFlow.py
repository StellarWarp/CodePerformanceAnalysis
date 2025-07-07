import json



def read_workflow_json_file(json_file):
    with open(json_file,'r',encoding='utf-8') as json_file:
        return json.load(json_file)