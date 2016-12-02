import numpy as np
import os
import re
import csv

# path1='./dataset-fb-valence-arousal-anon.txt'
# path2='./text_emotion.txt'
# fp1=open(path1,"r",errors="ignore")
# fp2=open(path2,"r",errors="ignore")
# text1=[]
#
# patho='./outtxt/'
# if not os.path.exists(patho):
#     os.mkdir(patho)
#
# for line in fp1.readlines():
#     text_line=line.strip('\n').split(',')[0]
#     with open('./outtxt/outtxt1.txt','a') as f1:
#         f1.write('\t')
#         f1.write(text_line)
#         f1.write('\t\n')
#
# lines = open("./outtxt/outtxt1.txt").readlines()
# del lines[0]
# open("./outtxt/outtxt1.txt","w").writelines(lines)
#
# text2=[]
# label2=[]
# for line in fp2.readlines():
#     text_line3=line.strip('\n').split(',')[3]
#     #text2.append(text_line3)
#     text_line1=line.split(',')[1]
#     #label2.append(text_line1)
#     with open('./outtxt/outtxt2.txt','a') as f2:
#         f2.write('\t')
#         f2.write(text_line3)
#         f2.write('\t')
#         f2.write(text_line1)
#         f2.write('\n')
#
# lines1 = open("./outtxt/outtxt2.txt").readlines()
# del lines1[0]
# open("./outtxt/outtxt2.txt","w").writelines(lines1)
#
# for fname in os.listdir('./outtxt/'):
#     for line in open(os.path.join('./outtxt/',fname),errors="ignore"):
#         line = re.sub(r'\b\"', '', line)        #there is no word on the left of "
#         line = re.sub(r'\"\b', '', line)
#         with open('./outtxt/outtxt3.txt','a',errors="ignore") as f3:
#             f3.write(line)
#
# path3='./outtxt/outtxt3.txt'
# fp3=open(path3,"r",errors="ignore")
# for line in fp3.readlines():
#     line=line.replace('"','')
#     with open('./outtxt/outtxt4.txt','a',errors="ignore") as f4:
#         f4.write(line)

for fname in os.listdir('./rawtwitterfeeds/'):
    lines=open(os.path.join('./rawtwitterfeeds/',fname),"r",errors="ignore").readlines()
    del lines[0]
    for line in open(os.path.join('./rawtwitterfeeds/',fname),errors="ignore"):
        line=line.strip('\n').split(',')
        if len(line)>=6:
            with open('./outtxt/outtxt_rtf.txt','a',errors="ignore") as f5:
                f5.write(line[5])
                f5.write('\n')

# for fname in os.listdir('D:/UIUC/studying/ECE544/544project/544_project_local/Code/rawtwitterfeeds/'):
#     with open(os.path.join('D:/UIUC/studying/ECE544/544project/544_project_local/Code/rawtwitterfeeds',fname),'r',errors="ignore") as csvfile:
#         reader=csv.DictReader(csvfile)
#         column=[row['text'] for row in reader if row]
#     with open('./outtxt/outtxt_rtf.txt','a',errors="ignore") as f5:
#         for i in range(len(column)):
#             f5.write('\t')
#             f5.write(column[i])
#             f5.write('\t\n')

# with open('A.csv','rb') as csvfile:
#     reader = csv.DictReader(csvfile)
#     column = [row['Age'] for row in reader]
# print column


