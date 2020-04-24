import os
import os.path as osp
import csv

def ReadAnnotation(csvName):
    annotaion_list = []
    with open(csvName,'r') as cvsfile:
        files = csv.reader(cvsfile)

        for file in files:
            img_path = file[0]
            bbox = []
            labels = []
            bbox.append(int(file[1]))
            bbox.append(int(file[2]))
            bbox.append(int(file[3]))
            bbox.append(int(file[4]))
            labels.append(img_path)
            labels.append(bbox)
            annotaion_list.append(labels)


            #print(file)
        print(files.line_num)
        cvsfile.close()
        return annotaion_list


if __name__ == '__main__':
    curDir = os.path.realpath(__file__)
    dirPath,fileName = os.path.split(curDir)
    csvName = osp.join(dirPath ,  '..' , 'Detect_Face.csv')

    if osp.exists(csvName):
        ReadAnnotation(csvName)
    else:
        print(csvName)
        print('csvName is error !')