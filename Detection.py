import cv2 #görüntü işleme kütüphanesi.
import math #matematiksel işlemler için fonksiyonlar içeren kütüphane.
import time #zamanla ilgili işlemleri gerçekleştirmek için kullanılan kütüphane.
import argparse #komut satırından argümanları işlemek için kullanılan kütüphane.

def getFaceBox(net,frame,conf_threshold=0.75):
    #Giriş çerçevesini kopyala
    frameOpencvDnn=frame.copy()

    #Çerçevenin yüksekliği ve genişliği
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]

    #Giriş çerçevesinden bir blob oluştur
    blob=cv2.dnn.blobFromImage(frameOpencvDnn,1.0,(300,300),[104,117,123],True,False)

    #Derin öğrenme modeline blob'u iletilir
    net.setInput(blob)

    # Modelin çalıştırılması
    detections=net.forward()

    #Tespit edilen yüz kutularını depolamak için boş bir liste oluşturuldu.
    bboxes=[]

    #Tespitler üzerinde döngü
    for i in range(detections.shape[2]):

         #Tespitin güvenilirlik puanı
         confidence=detections[0,0,i,2]

         # Güvenilirlik puanı belirli  güvenlik eşiği değerinden büyükse
         if confidence > conf_threshold:
             #Yüz kutusunun koordinatları (normalleştirilmiş)
             x1 = int(detections[0, 0, i, 3]*frameWidth)
             y1 = int(detections[0, 0, i, 4]*frameHeight)
             x2 = int(detections[0, 0, i, 5]*frameWidth)
             y2 = int(detections[0, 0, i, 6]*frameHeight)

             #Yüz kutusunu bboxes listesine ekle
             bboxes.append([x1,y1,x2,y2])

             # Yüz kutusunu çerçeve üzerine çizme işlemi
             cv2.rectangle(frameOpencvDnn,(x1,y1),(x2,y2),(0,255,0),int(round(frameHeight/150)),8)

    #Sonuç olarak, çerçeve üzerindeki yüz kutuları ve çerçeve döndürülür
    return frameOpencvDnn,bboxes

faceProto="opencv_face_detector.pbtxt"  #OpenCV'nin yüz tespiti modelinin prototxt dosyasının (konfigürasyon dosyası) dosya yolu.
faceModel="opencv_face_detector_uint8.pb" #Yüz tespiti için önceden eğitilmiş model dosyasının dosya yolu.

ageProto="age_deploy.prototxt" #Yaş tahmini için prototxt dosyasının dosya yolu.
ageModel="age_net.caffemodel" #Yaş tahmini için önceden eğitilmiş model dosyasının dosya yolu.

genderProto="gender_deploy.prototxt" #OpenCV'nin cinsiyet sınıflandırma modelinin prototxt dosyasının dosya yolu.
genderModel="gender_net.caffemodel" #Cinsiyet sınıflandırması için önceden eğitilmiş model dosyasının dosya yolu.

MODEL_MEAN_VALUES=(78.42633377603,87.7689143744,114.895847746) #Giriş görüntüsünü ön işleme için kullanılan ortalama değerler.
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(37-43)', '(48-53)', '(60-100)'] #Modelin sınıflandırdığı yaş aralıklarının listesi.
genderList=['ERKEK','KADIN']#Modelin sınıflandırdığı cinsiyet aralıklarının listesi.


#Yaş,Cinsiyet ve Yüz Modellerinin Yüklenmesi.
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)
faceNet=cv2.dnn.readNet(faceModel,faceProto)

#Kameranın başlatılması
cap=cv2.VideoCapture(0)

#Kenar boşluğunun piksel olarak değeri
padding=20

#Klavyeden bir tuşa basılana kadar döngüyü sürdür.
while cv2.waitKey(1)<0:

    #Başlangıç zamanını kaydeder.
    t=time.time()

    #Video akışından bir kare okur.
    hasFrame,frame=cap.read()

    #Eğer kare başarıyla okunamazsa, bekleyip döngüyü kır.
    if not hasFrame:
        cv2.waitKey()
        break

    #Kareyi yarı boyutuna küçült.
    small_frame=cv2.resize(frame,(0,0),fx=0.5,fy=0.5)

    #Küçültülmüş karede yüz tespiti yapar.
    frameFace,bboxes=getFaceBox(faceNet,small_frame)

    #Eğer yüz tespit edilemezse ekrana mesaj yazdır ve döngüye devam et.
    if not bboxes:
        print("Yüz Algılanamadı...")
        continue

    #Her tespit edilen yüzde işlem sağlanması için döngü oluşturuldu.
    for bbox in bboxes:

        #Yüz bölgesini belirle ve bir blob oluştur.
        face = small_frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        #Cinsiyet tahmini yapar ve sonucu ekrana yazdırır.
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print("Gender:{},conf={:.3f}".format(gender,genderPreds[0].max()))

        #Yaş tahmini yapar ve sonucu ekrana yazdırır.
        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print("Age Output: {}".format(agePreds))
        print("Age: {},conf={:.3f}".format(age,agePreds[0].max()))

        #Cinsiyet ve yaş  bilgilerini etiketler ve kara üzerinde yazdırır.
        label="{},{}".format(gender,age)
        cv2.putText(frameFace,label,(bbox[0],bbox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)

        #işlenmiş kareyi gösterir.
        cv2.imshow("Age Gender Demo",frameFace)

    #Geçen süreyi yazdırır.
    print("time : {:.3f}".format(time.time()-t))

