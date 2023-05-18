import pickle
import re
import matplotlib.pyplot as plt


def clean(x):
    #x=re.sub(r'\W',' ',x)
    #x = re.sub(r'[^a-zA-Z]',' ',x)
    x = re.sub("wouldn\'t",'would not',x)
    x = re.sub("they \ 've",'they have',x)
    
    #to remove html tags
    x = re.sub(r'<.*?>', '', x)
    
    #to remove everything except alpha
    x = re.sub(r'[^a-zA-Z]',' ',x)
     
    x = re.sub(r'\s+',' ',x)          #remove extra space's
    return x.lower()

n=int(input("Enter number of comments: "))
inp=[]
for i in range(n):
    inp.append(input("Enter statement "+str(i+1)+"\n"))

f=[]
for i in inp:
    s=clean(i)
    f.append(s)

with open("tokenizer.pkl",'rb') as f1:
    cv1=pickle.load(f1)
t=cv1.transform(f).toarray()

with open("predict_svm.pkl",'rb') as f1:
    svm=pickle.load(f1)

pred=svm.predict(t)
print(pred)
ze=0
one=0

for i in pred:
    if i==0:
        ze+=1
    else:
        one+=1
    
        
cnt=[ze,one]       
review= ['negative','positive']
plt.bar(review,cnt,width=1)
plt.show()