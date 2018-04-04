import json
file = open('test.json','r',encoding='utf-8')
rse=json.load(file)

with open('result.json','w') as fw:
	for line in rse:
		fw.write(str(line)+'\n')
# for r in rse:
# 	print(r)
print("end")
