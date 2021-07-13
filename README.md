# CNN-algorithm-Head-CT-images

Head CT görüntüleri üzerinde CNN algoritmasını python platformunda uygulanmıştır. Veriseti Kaggle üzerinden "Head CT - hemorrhage" başlığı ile bulunan https://www.kaggle.com/felipekitamura/head-ct-hemorrhage linkinden elde edilmiştir.

VERİSETİ:  HEAD CT - HEMORRHAGE 

Kullanılan veri seti, 200 adet .png uzantılı görüntü içermektedir. Bilgisayarlı tomografi kullanarak elde edilmiş insan kafa görüntüleri bulunmaktadır. Bunlardan 0-99 arası görüntüler hasta, 100-199 arası görüntüler sağlıklı insan görüntüleridir. “labels.csv” adlı dokümanda (id , hemorrage) bilgisi bulunmaktadır. Buradaki id, görüntülerin index numaralarını belirtirken; hemorrage ise hasta veya sağlıklı görüntü bilgisi taşıyan 0 ve 1 değerlerinden oluşmaktadır. 0 sayısı sağlıklı; 1 sayısı hasta anlamına karşılık gelmektedir.

VERİ ÖNİŞLEME
Labels.csv dosyasını ve 200 adet görüntü okutulduktan sonra, her görüntünün boyutlarının (şekillerinin) aynı olmadığı görülmüştür. Bu sebeple, cv2 kütüphanesi kullanılarak resize işlemi uygulanmıştır. Bunun sonucunda, kurulacak modelde kullanılmak üzere 200 adet görüntü 256x256x3 boyutlarında düzenlenmiştir.
Train verileri için, veri setindeki 200 görüntünün %75’i alınarak 150 adet görüntü; test veri seti için %10’u olacak şekilde 20 adet görüntü; validation veri seti için, %15’i olacak şekilde 30 adet görüntü birbirlerinden ayrılmıştır.

PROJENİN AMACI
Bu projedeki amaç, veri setindeki 200 adet görüntüyü python ortamında işledikten sonra, CNN algoritması kullanarak görüntüler üzerinde yüksek accuracy değeri ile hasta/sağlıklı sınıflamasını yapmaktır

SONUÇ
Grafik 1,  20 ayrı epoch sonucunda train aşamasındaki accuracy değerlerini göstermektedir. Train ve validation accuracy değerleri grafikte tutarlı olduğundan dolayı,  bu projede oluşturulan model de tutarlı denilebilir. Projede oluşturulan modelin tutarlılığından emin olmak için bir diğer bakılması gereken husus Grafik 2 olmalıdır.
Grafik 2, 20 ayrı epoch sonucunda oluşan train ve validation loss değerlerini göstermektedir. Bu grafikte train loss değeri azalırken validation loss değerinde de benzer şekilde azalma olmaktadır. Bu durum da oluşturulan modelin tutarlı olduğunu göstermektedir. Eğer, train loss değeri azalırken, validation loss değeri artış gösterse idi, bu durumda oluşturulan modelde overfitting adı verilen ezberleme/aşırı öğrenme gerçekleşmiş olurdu. Model, overfitting durumunda iken test  aşamasında düşük accuracy değeri görülmesine neden olmaktadır.
Test_pred değişkeni içerisinde 20 adet test görüntüsü için oluşturulan model tarafından üretilen tahminler bulunmaktadır. Bu tahmin değerleri [0,1] aralığındadır. Eğer yapılan tahmin 0.5 değerinden daha büyük ise “hasta” olarak sınıflama yapılmıştır aksi takdirde “sağlıklı” sınıfında bulunmaktadır. Yapılan tüm işlemler sonucunda elde edilen accuracy değeri %95 olarak çıkmaktadır.   
