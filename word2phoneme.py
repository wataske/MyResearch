import re
import jaconv
i=1
text=[]
with open("dataset/onomatopev.csv", "r", encoding="utf-8") as f:#txtファイルを指定する
    length=sum(1 for line in f)
while i<length+1:
    with open("dataset/onomatopev.csv", "r", encoding="utf-8") as f:#txtファイルを指定する
        data=f.readlines()[i-1]
        i+=1
        result = jaconv.hiragana2julius(data)
        print(text)
        text.append(result)
with open("onomatope.csv", "a", encoding="utf-8") as of:#音素に変換したものをファイルに書き込み
    for d in text:        
        of.writelines("%s\n" % d[0:])     
