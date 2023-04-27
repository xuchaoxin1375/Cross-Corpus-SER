from datetime import datetime, timezone
import  pickle as pkl

# Get the current UTC time
utc_time = datetime.now(timezone.utc)

# Convert UTC time to local time
local_time = utc_time.astimezone()

# Format local time as a string suitable for a Windows file name
now_utc_field_str = local_time.strftime('%Y-%m-%d_%H-%M-%S')

# print(utc_filed_str)  # Output: 2023-04-26_12-30-45
# now_utc = datetime.now(timezone.utc)
# now_utc_str=str(now_utc)

def load_pickle_by_name(best_clf):
    with open(best_clf, 'rb') as f:
        bclf=pkl.load(f)
    return bclf

def dump_pickle_by_name(bclf_objs,pkl_file,tag_time=True):
    # pkl = "bclf.pkl"
    name_fields=pkl_file.split(".")[:-1]
    name="".join(name_fields)
    if tag_time:
        name+=f"@{now_utc_field_str}"
    name+=".pickle"
    with open(name,"wb") as f:
        pkl.dump(bclf_objs,f)