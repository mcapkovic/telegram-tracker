# -*- coding: utf-8 -*-

# import modules
import pandas as pd
import argparse
import json
import glob
import time
import os
import re

# import submodules
from tqdm import tqdm

# import local submodules
from utils import get_config_attrs
from utils.ai_utils import num_tokens_from_messages


'''

Attributes
'''
config_attrs = get_config_attrs()

'''

Arguments
'''

parser = argparse.ArgumentParser(description='Arguments.')
parser.add_argument(
    '--data-path',
    '-d',
    type=str,
    required=False,
    help='Path where data is located. Will use `./output/data` if not given.'
)

# Parse arguments
args = vars(parser.parse_args())

# get main path
if args['data_path']:
    main_path = args['data_path']
    if main_path.endswith('/'):
        main_path = main_path[:-1]
else:
    main_path = './output/data'

# log results
text = f'''
Init program at {time.ctime()}

'''

# Collect JSON files
json_files_path = f'{main_path}/**/*_messages.json'
json_files = glob.glob(
    os.path.join(json_files_path),
    recursive=True
)

example_messages = {}

# Save dataset
msgs_file_path = f'{main_path}/msgs_dataset.csv'

# JSON files
for f in json_files:
    '''

    Iterate JSON files
    '''
    #  Get channel name
    username = f.split('.json')[0].replace('\\', '/').split('/')[-1].replace(
        '_messages', ''
    )

    # Echo
    print(f'Reading data from channel -> {username}')

    # read JSON file
    with open(f, encoding='utf-8', mode='r') as fl:
        obj = json.load(fl)
        fl.close()

    '''

	Reading posts
	'''
    messages = obj['messages']
    pbar = tqdm(total=len(messages))
    pbar.set_description(f'Reading posts')

    # main object
    response = {
        'channel_name': username
    }

    user_messages = []

    for idx, item in enumerate(messages):
        '''

        Iterate posts
        '''
        if item['_'] == 'Message':
            if item['message'] != '':
                message = item['message']
                message = message.replace('\n\n', '\n')
                message = message.replace('\n', 'NewLine')
                message = re.sub('\W+',' ',message).strip()
                message = message.replace('NewLine', '\n')
                user_messages.append(
                    {
                        'messages': [ 
                            { 
                                "role": "system",
                                "content": config_attrs['system_content']
                            },
                            {
                                "role": "user",
                                "content": config_attrs['user_content']
                            },
                            {
                                "role": "assistant",
                                "content": message
                            }
                        ]
                    },
                )

        # Update pbar
        pbar.update(1)

    # store messages
    example_messages[username] = user_messages

    # Close pbar connection
    pbar.close()

    print('-- END --')
    print('')

# calculate tokens and save data
for username, examples in example_messages.items():
    print(f'Number of messages: {len(examples)}')

    print("gpt-3.5-turbo")
    count = 0
    for example in examples:
        count += num_tokens_from_messages(example['messages'], "gpt-3.5-turbo")

    # example token count from the function defined above
    print(f"{count} prompt tokens counted by num_tokens_from_messages().")

    # save data to json
    file_path = f'{main_path}/{username}/{username}_fine_tunning.json'
    json_str = json.dumps(
        examples,
        indent=2,
        ensure_ascii=False,
        separators=(',',':')
    )
    writer = open(file_path, mode='w', encoding='utf-8')
    writer.write(json_str)
    writer.close()