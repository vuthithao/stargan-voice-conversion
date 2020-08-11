import os

path = 'models_all/models'
di = os.listdir(path)
best = [200000, 322000, 480000, 800000, 330000]
for i in di:
    try:
        if int(i.split('-')[0]) not in best:
            os.remove(os.path.join(path, i))
    except:
        continue
