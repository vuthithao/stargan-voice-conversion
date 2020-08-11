import subprocess

# female
lop = [2, 3, 4, 5, 6, 7, 8, 9, 10]
inter = [330000, 200000, 550000, 500000, 100000]

for i in inter:
    for p in lop:
        subprocess.check_output("python convert.py --resume_iters {} --src_spk p0 --trg_spk {}".format(str(i), str(p)), shell=True)

print('female_done')
# male
lop = [2, 0, 4, 5, 6, 7, 8, 9, 10]
inter = [330000, 200000, 550000, 500000, 100000]

for i in inter:
    for p in lop:
        subprocess.check_output("python convert.py --resume_iters {} --src_spk p3 --trg_spk {}".format(str(i), str(p)), shell=True)