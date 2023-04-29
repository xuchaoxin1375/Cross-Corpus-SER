from datetime import datetime, timezone
import pickle as pkl


# Get the current UTC time
def get_now_utc_field_str():
    utc_time = datetime.now(timezone.utc)

    # Convert UTC time to local time
    local_time = utc_time.astimezone()

    # Format local time as a string suitable for a Windows file name
    now_utc_field_str = local_time.strftime("%Y-%m-%d_%H-%M-%S")
    return now_utc_field_str


now_utc_field_str = get_now_utc_field_str()


def load_pickle_by_name(pickle_file):
    with open(pickle_file, "rb") as f:
        bclf = pkl.load(f)
    return bclf


def dump_pickle_by_name(bclf_objs, pickle_file, tag_time=True):
    # pkl = "bclf.pkl"
    name_fields = pickle_file.split(".")[:-1]
    name = "".join(name_fields)
    if tag_time:
        name += f"@{now_utc_field_str}"
    name += ".pickle"
    with open(name, "wb") as f:
        pkl.dump(bclf_objs, f)
