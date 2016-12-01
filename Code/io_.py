import numpy as np
import os
import re

path1='./dataset-fb-valence-arousal-anon.txt'
path2='./text_emotion.txt'
fp1=open(path1,"r",errors="ignore")
fp2=open(path2,"r",errors="ignore")
text1=[]

patho='./outtxt/'
if not os.path.exists(patho):
    os.mkdir(patho)

for line in fp1.readlines():
    text_line=line.strip('\n').split(',')[0]
    with open('./outtxt/outtxt1.txt','a') as f1:
        f1.write('\t')
        f1.write(text_line)
        f1.write('\t\n')

lines = open("./outtxt/outtxt1.txt").readlines()
del lines[0]
open("./outtxt/outtxt1.txt","w").writelines(lines)

text2=[]
label2=[]
for line in fp2.readlines():
    text_line3=line.strip('\n').split(',')[3]
    #text2.append(text_line3)
    text_line1=line.split(',')[1]
    #label2.append(text_line1)
    with open('./outtxt/outtxt2.txt','a') as f2:
        f2.write('\t')
        f2.write(text_line3)
        f2.write('\t')
        f2.write(text_line1)
        f2.write('\n')

lines1 = open("./outtxt/outtxt2.txt").readlines()
del lines1[0]
open("./outtxt/outtxt2.txt","w").writelines(lines1)

for fname in os.listdir('./outtxt/'):
    for line in open(os.path.join('./outtxt/',fname)):
        line = re.sub(r'\b\"', '', line)        #there is no word on the left of "
        line = re.sub(r'\"\b', '', line)
        with open('./outtxt/outtxt3.txt','a') as f3:
            f3.write(line)




