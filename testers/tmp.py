##
from tqdm import tqdm
from time import sleep
from audio.core import best_estimators
ests=best_estimators()
ests=tqdm(ests)
for x in ests:
    sleep(0.5)
    print(x)
##
