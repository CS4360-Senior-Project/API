import csv
import os

""" CREATES THE TXT FOR MACHINE LEARNING """

#input_dir for dataset
input_dir = '../SeniorExperience/OriginalImages'
txt_file = "../SeniorExperience/API/config/api/backend/prompt_dataset.txt"

prompt1 = "../SeniorExperience/API/config/api/backend/prompts/prompt1.txt"
prompt2 = "../SeniorExperience/API/config/api/backend/prompts/prompt2.txt"
prompt3 = "../SeniorExperience/API/config/api/backend/prompts/prompt3.txt"

data = []

for file_name in os.listdir(input_dir):
    # extract the prompt
    file_name = file_name[:-4]
    prompt = file_name.split("_")[2][1:]

    if prompt == "LND":
        """ Write text to prompt1 files """
        with open(prompt1, 'r') as file:
            prompt1_text = file.read()
            write_string = str(file_name) + " ok # " + str(prompt1_text)
            data.append(write_string)
    
    if prompt == "PHR":
        """ Write text to prompt2 files """
        with open(prompt2, 'r') as file:
            prompt2_text = file.read()
            write_string = str(file_name) + " ok # " + str(prompt2_text)
            data.append(write_string)

    if prompt == "WOZ":
        """ Write text to prompt3 files """
        with open(prompt3, 'r') as file:
            prompt3_text = file.read()
            write_string = str(file_name) + " ok # " + str(prompt3_text)
            data.append(str(write_string))

print(data)

with open(txt_file, "w", newline="") as file:
    for string in data:
        file.write("%s\n" % string)