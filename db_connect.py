import pymysql.cursors
import cv2

def baglan(path,video,emotion):
    db = pymysql.connect(host='localhost',
                         user='root',
                         password='',
                         db='images',
                         charset='utf8mb4',
                         cursorclass=pymysql.cursors.DictCursor)

    baglanti = db.cursor()
    """if (baglanti):
        print("db bağlanti başarılı")
    else:
        print("db bağlanti başarısız")"""
    if(video == 'test_image/video1.mov'):
        sonuc = baglanti.execute('INSERT INTO tbl_giris VALUES(%s,%s,%s)', (None, path,emotion))
        db.commit()
        print("resim yolu girişe eklendi.")

    elif (video == 'test_image/video2.mov'):
        sonuc = baglanti.execute('INSERT INTO tbl_cıkıs VALUES(%s,%s,%s)', (None, path,emotion))
        db.commit()
        print("resim yolu cıkısa eklendi.")

