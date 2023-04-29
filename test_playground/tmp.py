d = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C':{"a":[7, 8, 9],"b":7}}
import json

def dict_to_filetag(d):
    # Convert dictionary to JSON string
    json_str = json.dumps(d)
    remove_chars=['"',' ']
    for c in remove_chars:
        json_str=json_str.replace(c,'')
    # Replace invalid characters with hyphen
    rep_dict={
        ":":"=",
        # '"':'',
        # "'":""
    }
    for char in json_str:
        if rep_dict.get(char):
            json_str = json_str.replace(char, rep_dict[char])
    # Truncate string if too long
    # max_len = 260
    # if len(json_str) > max_len:
    #     json_str = json_str[:max_len]

    return json_str


res=dict_to_filetag(d)
print(res)
