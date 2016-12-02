from __future__ import print_function
from operator import itemgetter
from collections import Counter
import gensim
import numpy as np
from data_pipeline import Sentences
import sys

def find_dictionaries(path,split_line,split_method,label):

    dic_sur_initial = {}
    dic_sad_initial = {}
    dic_joy_initial = {}
    dic_dis_initial = {}
    dic_fear_initial = {}
    dic_anger_initial= {}

    sentences = Sentences(path, split_line=split_line, split_method=split_method, label=label)
    countsur=0
    countsad=0
    countjoy=0
    countdis=0
    countfear=0
    countanger=0
    for line,label in sentences:
        if label=='surprise':
            countsur+=1
            for i in range(len(line)):
                if line[i] not in dic_sur_initial.keys():
                    dic_sur_initial.setdefault(line[i], 1)
                else:
                    dic_sur_initial[line[i]]+=1
        elif label=='sadness':
            countsad+=1
            for i in range(len(line)):
                if line[i] not in dic_sad_initial.keys():
                    dic_sad_initial.setdefault(line[i], 1)
                else:
                    dic_sad_initial[line[i]]+=1
        elif label=='joy':
            countjoy+=1
            for i in range(len(line)):
                if line[i] not in dic_joy_initial.keys():
                    dic_joy_initial.setdefault(line[i], 1)
                else:
                    dic_joy_initial[line[i]]+=1
        elif label=='disgust':
            countdis+=1
            for i in range(len(line)):
                if line[i] not in dic_dis_initial.keys():
                    dic_dis_initial.setdefault(line[i], 1)
                else:
                    dic_dis_initial[line[i]]+=1
        elif label=='fear':
            countfear+=1
            for i in range(len(line)):
                if line[i] not in dic_fear_initial.keys():
                    dic_fear_initial.setdefault(line[i], 1)
                else:
                    dic_fear_initial[line[i]]+=1
        elif label=='anger':
            countanger+=1
            for i in range(len(line)):
                if line[i] not in dic_anger_initial.keys():
                    dic_anger_initial.setdefault(line[i], 1)
                else:
                    dic_anger_initial[line[i]]+=1
    dic_sur={}
    sur_keys=list(dic_sur_initial.keys())
    sur_values=list(dic_sur_initial.values())
    for i in range(len(dic_sur_initial)):
        sur_values[i]/=countsur
        dic_sur[sur_keys[i]]=sur_values[i]

    dic_sad = {}
    sad_keys = list(dic_sad_initial.keys())
    sad_values = list(dic_sad_initial.values())
    for i in range(len(dic_sad_initial)):
        sad_values[i] /= countsad
        dic_sad[sad_keys[i]] = sad_values[i]

    dic_joy = {}
    joy_keys = list(dic_joy_initial.keys())
    joy_values = list(dic_joy_initial.values())
    for i in range(len(dic_joy_initial)):
        joy_values[i] /= countjoy
        dic_joy[joy_keys[i]] = joy_values[i]

    dic_dis = {}
    dis_keys = list(dic_dis_initial.keys())
    dis_values = list(dic_dis_initial.values())
    for i in range(len(dic_dis_initial)):
        dis_values[i] /= countdis
        dic_dis[dis_keys[i]] = dis_values[i]

    dic_fear = {}
    fear_keys = list(dic_fear_initial.keys())
    fear_values = list(dic_fear_initial.values())
    for i in range(len(dic_fear_initial)):
        fear_values[i] /= countfear
        dic_fear[fear_keys[i]] = fear_values[i]

    dic_anger = {}
    anger_keys = list(dic_anger_initial.keys())
    anger_values = list(dic_anger_initial.values())
    for i in range(len(dic_anger_initial)):
        anger_values[i] /= countanger
        dic_anger[anger_keys[i]] = anger_values[i]



    dic_c=[]
    dic_c.append(dic_sur)
    dic_c.append(dic_sad)
    dic_c.append(dic_joy)
    dic_c.append(dic_dis)
    dic_c.append(dic_fear)
    dic_c.append(dic_anger)         #dictionaries for every category not sorted


    dic_sur= sorted(dic_sur.items(), key=itemgetter(1), reverse = True)
    dic_sad= sorted(dic_sad.items(), key=itemgetter(1), reverse = True)
    dic_joy= sorted(dic_joy.items(), key=itemgetter(1), reverse = True)
    dic_dis= sorted(dic_dis.items(), key=itemgetter(1), reverse = True)
    dic_fear= sorted(dic_fear.items(), key=itemgetter(1), reverse = True)
    dic_anger= sorted(dic_anger.items(), key=itemgetter(1), reverse = True)

    #dic_sur is a list

    dic=[]
    dic.append(dic_sur)
    dic.append(dic_sad)
    dic.append(dic_joy)
    dic.append(dic_dis)
    dic.append(dic_fear)
    dic.append(dic_anger)           #list for every category sorted

    return dic_c,dic


def print_most_occurrence(dic):
    print('most occurence when label= surprise ')
    for i in range(10):
        print(dic[0][i])

    print('----------------------------------')
    print('most occurence when label= sadness ')
    for i in range(10):
        print(dic[1][i])

    print('----------------------------------')
    print('most occurence when label= joy ')
    for i in range(10):
        print(dic[2][i])

    print('----------------------------------')
    print('most occurence when label= disguist ')
    for i in range(10):
        print(dic[3][i])

    print('----------------------------------')
    print('most occurence when label= fear ')
    for i in range(10):
        print(dic[4][i])

    print('----------------------------------')
    print('most occurence when label= anger ')
    for i in range(10):
        print(dic[5][i])


def find_common_dictionary(dic,dic_c):
    #---------------find the common dictionary-----------------#

    dic_label=['surprise', 'sadness', 'joy', 'disgust', 'fear', 'anger']

    dic_common={}


    for i in range(len(dic)):
        dic_common=dict(Counter(dic_common)+Counter(dic_c[i]))
    #print(sorted(dic_common.items(),key=lambda dic_common:dic_common[1],reverse=True))

    return dic_common, dic_label


def find_most_repres(dic,dic_common,dic_label):
    #------------print the most representative words-------------#

    for j in range(len(dic)):
        dic_p={}    #dic_p is the dictionary that is to be sorted
        dic_count_sorted= dic[j] #dic_count_sorted is a list which contains sorted list of every category  with number of occurrence
        l=len(dic_count_sorted)
        medium=int(l/12)
        for k in range(len(dic_count_sorted)):
            occurrence_p=float(dic_count_sorted[k][1]/dic_common[dic_count_sorted[k][0]])
            word=dic_count_sorted[k][0]
            dic_p[word]=occurrence_p
            # dic_tobesort=sorted(dic_tobesort, key=lambda dic_tobesort: dic_tobesort[1], reverse=True)
        dic_prob_sorted=sorted(dic_p.items(),key=lambda dic_p:dic_p[1],reverse=True)
        #print(dic_prob_sorted)
        print('-----------------------------------')
        print('most representative words for '+dic_label[j])
        count=0
        for i in range(len(dic_prob_sorted)):
            word=dic_prob_sorted[i][0]
            for j in range(len(dic_count_sorted)):
                if word==dic_count_sorted[j][0]:
                    number=dic_count_sorted[j][1]
            if (number>=dic_count_sorted[medium][1]):
                print(dic_prob_sorted[i][0])
                count+=1
            if count>9:
                break
    #restrict the number of times that a word to be sort that occurs in a dictionary


def find_least_repres(dic_c,dic_common):
    #----------------print the least representative words-----------------#
    keys=list(dic_common.keys())
    values=list(dic_common.values())
    sumtotal=0
    for i in range(len(values)):
        sumtotal+=values[i]
    dic_var={}
    dic_common_sorted=sorted(dic_common.items(),key=lambda dic_common:dic_common[1],reverse=True)
    l=len(dic_common_sorted)
    var_m=[]
    dic_final={}
    for i in range(len(dic_common)):
        pm=[]
        for j in range(len(dic_c)):
            if keys[i] in dic_c[j]:
                pm.append(dic_c[j][keys[i]]/values[i])
            else:
                pm.append(0)
        pmarray1=np.array(pm)
        sum1=pmarray1.sum()
        pmarray2=pmarray1*pmarray1
        sum2=pmarray2.sum()
        mean=sum1/(len(dic_c))
        var=sum2/(len(dic_c))-mean**2
        word=keys[i]
        dic_var[word]=var
        #var_m.append(var)
        score=-np.log((values[i]/sumtotal))*var
        dic_final[word]=score
    dic_final_sorted=sorted(dic_final.items(),key=lambda dic_final:dic_final[1])
    print('-----------------------------')
    print('print the least representative words')
    count=0
    # print(dic_common_sorted[:10])
    for i in range(100):
        sys.stdout.write('\" '+dic_final_sorted[i][0]+' \"' + ',')
    # print(dic_final_sorted[:100])
    # for i in range(len(dic_final_sorted)):
        # if dic_common[dic_final_sorted[i][0]]>=dic_common_sorted[int(100)][1]:
        #     print(dic_final_sorted[i], dic_common[dic_final_sorted[i][0]])
        #     count+=1
        # if count>9:
        #     break


#--------------------------------------main-------------------------------------#
if __name__=="__main__":

    dic_c,dic=find_dictionaries(path='./data_set/redu/',split_line=True, split_method='Twitter',label=True)

    print_most_occurrence(dic)

    dic_common, dic_label=find_common_dictionary(dic,dic_c)

    # find_most_repres(dic,dic_common,dic_label)

    find_least_repres(dic_c,dic_common)





