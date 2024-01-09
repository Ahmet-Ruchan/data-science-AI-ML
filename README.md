# FIFA 23: OYUNCU PERFORMANSININ SIRLARI

## A.1. BİRİNCİ ANALİZ ÖĞESİ

Futbol, dünya genelinde büyük bir tutkudur ve FIFA serisi, futbolseverlerin oyuncularla interaktif bir şekilde etkileşime geçmelerini sağlayan bir platform sunar. FIFA 23 oyunundaki oyuncu verilerini inceleyerek, oyuncu performansının ne şekilde şekillendiğini ve bu performansın oyun içindeki etkilerini anlamaya çalışacağım. Oyuncuların yetenekleri, pozisyonları ve diğer özellikler üzerinden bir hikaye çıkarmaya hazır mısınız?

* Oyuncu Performansı ve Özellikleri:
FIFA 23 veri seti, oyuncuların yeteneklerini, pozisyonlarını ve diğer çeşitli özelliklerini içerir. Her bir oyuncu, sahada nasıl performans gösterdiğini belirten özelliklere sahiptir. Bu veri seti, futbolun gerçek dünyasındaki oyuncuların sanal performanslarını analiz etmek için bir fırsat sunuyor.

* Veri Seti İncelemesi:
Veri setindeki her bir satır, bir oyuncunun özelliklerini ve performansını temsil eder. Yetenek puanları, oyun içi konumları ve diğer kritik bilgileri içeren bu veriler, oyuncuların hangi koşullarda en iyi performansı sergilediğini anlamamıza yardımcı olacak. Bu bölümde, oyuncuların performansını etkileyen ana faktörleri belirlemeye odaklanacağım.
Veri Ön İşleme:
Veri setini analiz etmeden önce, eksik verileri kontrol edecek, aykırı değerleri ele alacak ve gerekirse kategorik değişkenleri sayısallaştıracağım. Bu adımlar, veri setini sağlıklı bir şekilde analiz etmek için gereklidir.

* Sonuç:
Bu makalede, FIFA 23 oyunundaki oyuncu verilerini kullanarak, oyuncu performansının arkasındaki sırları çözmeye çalıştım. Veri setini inceledikten ve ön işleme adımlarını uyguladıktan sonra, bir sonraki adım olarak oyuncu performansını etkileyen faktörleri belirleyecek ve bir regresyon modeli oluşturarak oyuncu performansını tahmin etmeye çalışacağım.
Bu çalışma, futbolseverlere oyuncuların performansını daha derinlemesine anlama ve oyun içinde stratejik kararlar vermelerine yardımcı olma fırsatı sunabilir. İlerleyen süreçte, regresyon analizi sonuçlarını değerlendirecek ve oyuncu performansının oyun içindeki etkilerini daha ayrıntılı bir şekilde inceleyeceğim.
Bu raporda belirli metrikler çerçevesinde, futbolcunun ‘Bitiricilik’ (Finishing)  özelliğini tahmin etmeye çalışacağız. Öncelikle kütüphaneleri yükleyerek başlıyoruz ve devamında şekil 1.0’ da görüldüğü gibi ilk beş veriyi ekrana yazdırıyoruz. Bu kod bloğu, 'player_stats.csv' adlı bir veri setini okuyarak, bu veriyi bir DataFrame'e dönüştürüyor. pd.read_csv fonksiyonu, bir CSV dosyasını okuyup veri çerçevesine dönüştürmek için kullanılır. encoding='latin-1' parametresi, dosyanın Latin-1 karakter setini kullanmasını sağlar ve olası karakter hatalarını önler. Bu aşamada, projenin geri kalanında kullanılacak olan temel veri setini yükledik. Veri setini daha yakından inceleyerek, içerdiği sütunlar, veri tipleri ve genel istatistiksel bilgiler gibi unsurları anlamak, bir sonraki adıma geçiş için önemlidir. Bu inceleme adımı, veri setinin temel özelliklerini anlamamıza yardımcı olacak ve projenin ilerleyen aşamalarında nasıl manipüle edileceğini belirleyecektir.
Bu veri setimizde, model eğitmek için kullanabileceğimiz sütunlar haricindeki diğer sütunları veri setimizden şekil 1.1’ de çıkartıyoruz. Şekil 1.2’de veri setimizin genel bilgilerini ve metriklerini gözlemliyoruz.

## 1. ÇOKLU REGRESYON İLE TAHMİN ETME

### Şekil 1.3’ te çoklu regresyona başlamak için tahmin etmek istediğimiz özelliğimiz olan finishing (bitiricilik) özelliğini y değişkenine, geriye kalan özellikleri de X değişkenine atıyoruz. 
### Şekil 1.4: Bu kod bloğu, train_test_split fonksiyonunu kullanarak bağımsız değişkenleri (X) ve bağımlı değişkeni (y) belirtilen oranda eğitim ve test kümelerine böler. test_size=0.20 parametresi, veri setinin yüzde 20'sinin test kümesi olarak ayrılacağını belirtir. random_state=12345 parametresi ise, veri setinin bölünmesindeki rastgele durumu kontrol eder ve aynı bölme işleminin tekrarlanabilir olmasını sağlar.
Eğitim ve test kümelerinin ayrılması, modelin genel performansını değerlendirmek ve aşırı uyum (overfitting) gibi sorunları önlemek için önemlidir. Bu aşamadan sonra, model eğitimi ve değerlendirmesi için hazır durumda olan veri kümeleri elde edilmiş olacaktır. Bu kod bloğu, LinearRegression sınıfını kullanarak bir lineer regresyon modeli oluşturur. Lineer regresyon, bağımlı değişkenin bağımsız değişkenlerle lineer bir ilişkisi olduğu durumlarda kullanılır. Modelin amacı, veri setindeki bu ilişkiyi temsil eden bir doğruyu öğrenmek ve bu doğruyu kullanarak tahminlerde bulunmaktır.
Oluşturulan model, daha önce belirlenen eğitim kümesi (X_train, y_train) üzerinde eğitilecektir. Modelin eğitilmesi, lineer regresyonun veri setine uygun bir şekilde adapte olmasını ve gelecekteki tahminlerde doğru sonuçlar üretmesini sağlar. Eğitildikten sonra, model test kümesi (X_test) üzerinde değerlendirilecek ve performans metrikleri kullanılarak başarısı ölçülecektir. Model oluşturulduktan sonra modeli eğitmeye başlıyoruz. Eğitimin ardından y_predict değişkenine atadığımız bir tahmin işlemi gerçekleştiriyoruz.

### Şekil 1.5:

* R^2 Score (R Kare Skoru):
R^2 skoru, modelin bağımsız değişkenler tarafından açıklanan toplam varyansın yüzde kaçını açıkladığını gösterir. 1'e ne kadar yakınsa, modelin daha iyi performans gösterdiğini gösterir.

* RMSE (Root Mean Squared Error - Karekök Ortalama Hata):
RMSE, modelin gerçek değerlerle tahmin değerleri arasındaki ortalama hata miktarını ölçer. Daha düşük RMSE değerleri, modelin daha iyi tahminler yaptığını gösterir.

* MAE (Mean Absolute Error - Ortalama Mutlak Hata):
MAE, modelin gerçek değerlerle tahmin değerleri arasındaki mutlak hataların ortalamasını ölçer. Daha düşük MAE değerleri, daha iyi bir model performansını işaret eder.

* MSE (Mean Squared Error - Ortalama Kare Hata):
MSE, modelin gerçek değerlerle tahmin değerleri arasındaki kare hataların ortalamasını ölçer. RMSE'nin karesi olduğu için, bu iki metrik birbirine benzerdir.
Bu metrikler, modelin başarısını farklı açılardan değerlendirmemize olanak tanır. Özellikle R^2 skoru, modelin genel performansını anlamak için kullanılırken, RMSE, MAE ve MSE gibi metrikler modelin hata oranları hakkında daha ayrıntılı bilgiler sağlar. Bu metriklerin sonuçları, modelin belirli bir veri setinde ne kadar iyi performans gösterdiğini anlamamıza yardımcı olacaktır.

### Şekil 1.6:
Bu kod bloğu, gerçek değerlerin (y_test) ve modelin tahmin ettiği değerlerin (y_predict) birbirine görsel olarak nasıl benzediğini gösteren bir scatter plot grafiği oluşturur. Her bir nokta, bir gözlem birimini temsil eder. Eğik kırmızı çizgi, mükemmel bir tahmin durumunu temsil eder; yani, gerçek değerler ile tahmin edilen değerler arasında bir fark olmazsa bu çizgi üzerinde olurlardı.
Grafik, noktaların ne kadar yakın olduğunu ve eğik çizginin ne kadar yakın olduğunu göstererek modelin ne kadar başarılı olduğunu değerlendirmemize yardımcı olur. İdeal durumda, noktalar eğik çizgi üzerinde bir araya gelir. Eğer noktalar eğik çizginin etrafında toplanmışsa, modelin iyi tahminler yaptığı söylenebilir.

## 2. CROSS VALIDATION (ÇAPRAZ DOĞRULAMA)  İLE TAHMİN ETME

### Şekil 1.7:

* R^2 Score (Çapraz Doğrulama):
Modelin genel performansını ölçen R^2 skorunun çapraz doğrulama sonuçlarının ortalaması.

* MAE (Negatif Ortalama Mutlak Hata):
Çapraz doğrulama kullanılarak modelin negatif ortalama mutlak hata değerlerinin ortalaması.

* MSE (Negatif Ortalama Kare Hata):
Çapraz doğrulama kullanılarak modelin negatif ortalama kare hata değerlerinin ortalaması.

* RMSE (Negatif Kök Ortalama Kare Hata):
Çapraz doğrulama kullanılarak modelin negatif kök ortalama kare hata değerlerinin ortalaması.
Bu metrikler, modelin çapraz doğrulama ile genel bir performansını değerlendirmek için kullanılır. Negatif değerlerin kullanılmasının nedeni, sklearn kütüphanesinde genellikle hataların küçük olması durumunda daha iyi olduğu kabul edildiğinden, bu değerleri azaltma amacıdır. Çapraz doğrulama sonuçları, modelin farklı veri kesimleri üzerinde nasıl performans gösterdiğini daha güvenilir bir şekilde değerlendirmemizi sağlar.

### Şekil 1.8:
Bu grafik, çapraz doğrulama sonuçlarından elde edilen mavi renkteki noktaları içermektedir. Bu noktalar, gerçek değerlerle modelin tahmin ettiği değerleri gösterir. Eğik yeşil çizgi, mükemmel bir tahmin durumunu temsil eder; yani, gerçek değerler ile tahmin edilen değerler arasında bir fark olmazsa bu çizgi üzerinde olurlardı.
Grafik, modelin farklı veri kesimleri üzerinde nasıl performans gösterdiğini daha net bir şekilde gösterir. Eğer noktalar eğik çizginin etrafında toplanmışsa, modelin genelde iyi tahminler yaptığı söylenebilir. Ancak, geniş bir dağılım söz konusu ise, modelin bazı durumlarda daha zayıf tahminler yaptığı anlaşılabilir.

## 3. GRID SEARCH  İLE TAHMİN ETME

#### Şekil 1.9:

Bu kod bloğu, Grid Search yöntemi kullanılarak lineer regresyon modelinin hiperparametrelerini değerlendirir. parameters sözlüğü, değerlendirilecek hiperparametre kombinasyonlarını içerir. Skorlama fonksiyonları, modele uygulanan çapraz doğrulama sonuçlarını değerlendirmek için kullanılır.
Çalışma sonucunda, en iyi skorlar ve bu skorlara karşılık gelen hiperparametre kombinasyonları yazdırılır. En iyi model, Grid Search tarafından seçilen en iyi hiperparametrelerle oluşturulur. Elde edilen en iyi model ile çapraz doğrulama kullanılarak tahminler yapılarak, modelin genel performansı daha ayrıntılı bir şekilde değerlendirilir.

### Şekil 2.0:
Bu kod bloğu, Grid Search ile iyileştirilmiş modelin çapraz doğrulama sonuçlarından elde edilen tahminleri görselleştiren bir scatter plot grafiği oluşturur. Sarı renkteki noktalar, gerçek değerlerle bu modelin tahmin ettiği değerleri temsil eder. Eğik açık mavi çizgi, mükemmel bir tahmin durumunu temsil eder; yani, gerçek değerler ile tahmin edilen değerler arasında bir fark olmazsa bu çizgi üzerinde olurlardı.
Bu grafik, iyileştirilmiş modelin gerçek değerlere göre ne kadar doğru tahminler yaptığını daha iyi anlamamıza yardımcı olur. Eğer noktalar eğik çizginin etrafında toplanmışsa, modelin genelde iyi tahminler yaptığı söylenebilir. Ancak, geniş bir dağılım söz konusu ise, modelin bazı durumlarda daha zayıf tahminler yaptığı anlaşılabilir. Bu görselleştirme, iyileştirilmiş modelin performansını değerlendirmek için önemli bir araçtır.

## A.2. İKİNCİ ANALİZ ÖĞESİ

Bu bölümde veri setimizden belirli nitelikleri çıkararak daha spesifik bir yapı oluşturacağız. Doğrudan finishing (bitiricilik) ile bağlantısı olmayan sütunları çıkartarak daha kararlı bir yapı oluşturalım. 

## A.3. YENİ SENARYO
Futbolda bir forvette bulunması gereken en önemli özelliklerden: Top kontrolü, top sürme yeteneği, reaksiyon, kondisyon, güç, denge, koşu hızı, kafa vuruşu, şut gücü ve uzak mesafe şutu özellikleridir. Bu sebeple veri setimizden bunun dışındaki nitelikleri çıkartarak yeniden model eğiteceğiz ve tahmin yeteneğine bakacağız. Amacımız yine finishing (bitiricilik) niteliğini tahmin etmeye çalışmak olacaktır.

### Şekil 2.1:
Bu şemada eski veri setinden 4 değeri daha çıkartarak çok daha kararlı ve çok daha spesifik bir veri seti oluşturuyoruz. Daha sonrasında Şekil 2.2 ‘de modeli tekrar test ve train verileri olacak şekilde ayırıyoruz ve modelimizi oluşturuyoruz. Bütün her şey hazır şimdi sırada Grid Search ve Cross Validation ile model eğitmek var.

### Şekil 2.3:
Bu kod bloğu, Grid Search yöntemi kullanılarak lineer regresyon modelinin hiperparametrelerini değerlendirir. parameters sözlüğü, değerlendirilecek hiperparametre kombinasyonlarını içerir. Skorlama fonksiyonları, modele uygulanan çapraz doğrulama sonuçlarını değerlendirmek için kullanılır.

### Şekil 2.4:
Bu kod bloğu, Grid Search ile iyileştirilmiş modelin ikinci senaryo için gerçek değerlerle yaptığı tahminleri görselleştiren bir scatter plot grafiği oluşturur. Mavi renkteki noktalar, gerçek değerlerle bu modelin tahmin ettiği değerleri temsil eder. Eğik pembe çizgi, mükemmel bir tahmin durumunu temsil eder; yani, gerçek değerler ile tahmin edilen değerler arasında bir fark olmazsa bu çizgi üzerinde olurlardı. Bu grafik, iyileştirilmiş modelin ikinci senaryoda gerçek değerlere göre ne kadar doğru tahminler yaptığını daha iyi anlamamıza yardımcı olur. 
