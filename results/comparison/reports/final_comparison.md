# Nihai karsilastirma raporu

## Degerlendirme protokolu
- Denek-bagimli; ayni split manifestosu; test bir kez kullanildi.
- Katilimci sayisi: 50

## Toplu dogruluk (ortalama)
- csp_lda: ort=0.680, medyan=0.675
- eegnet: ort=0.549, medyan=0.519
- atcnet: ort=0.594, medyan=0.559

## Istatistik (accuracy)
- eegnet_vs_csp_lda: mean_diff=-0.130, holm_p=0.000, rank_biserial=-0.83, anlamli=True
- atcnet_vs_csp_lda: mean_diff=-0.085, holm_p=0.000, rank_biserial=-0.67, anlamli=True
- atcnet_vs_eegnet: mean_diff=+0.045, holm_p=0.003, rank_biserial=+0.52, anlamli=True

## Sinirliliklar
- Istatistiksel anlamlilik pratik/klinik degeri garanti etmez; etki buyuklugu ve maliyet yorumlanmali.
- Tek oturum; cevrimdisi; deneklerarasi degiskenlik yuksek.
- 'AI ustun' iddiasi yalnizca bir dogruluk/karmasiklik odunlesmesi sunuyor olabilir.

## Nihai sonuc
- En yuksek ortalama/medyan dogruluga sahip model ve tutarliligi yukaridaki tablolardan degerlendirilir.
- Hicbir sonuc, defterler gercekten calistirilmadan iddia edilemez.